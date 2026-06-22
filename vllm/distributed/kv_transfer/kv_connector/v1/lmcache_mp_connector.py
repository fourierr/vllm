# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import zmq
from lmcache.integration.vllm.utils import mla_enabled
from lmcache.utils import init_logger as lmcache_init_logger

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

try:
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
    )

    try:
        from lmcache.v1.multiprocess.custom_types import RequestAllocationRecord
    except ImportError:
        from lmcache.v1.multiprocess.custom_types import (
            BlockAllocationRecord as RequestAllocationRecord,
        )
except ImportError:
    from lmcache.v1.multiprocess.custom_types import (
        BlockAllocationRecord as RequestAllocationRecord,
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
    )

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = lmcache_init_logger(__name__)


# Helper functions
def reformat_block_ids(block_ids: tuple[list[int], ...] | None) -> list[int]:
    if block_ids is None:
        return []
    assert isinstance(block_ids, tuple), (
        f"Expected block_ids to be a tuple of lists, but got {type(block_ids)}"
    )

    if len(block_ids) > 1:
        raise RuntimeError(
            "LMCacheMPConnector only works without hybrid kv cache manager. "
            "Please pass --disable-hybrid-kv-cache-manager when starting vllm"
        )

    return block_ids[0]


def extract_world_size_and_kv_rank(
    world_size: int,
    rank: int,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """
    Convert the rank for the MLA.
    """
    use_mla = mla_enabled(vllm_config.model_config)
    if not use_mla:
        return world_size, rank
    else:
        # Tensor parallel does not change the KV caches for MLA models.
        # So we need to "exclude" the effect of TP on rank and world size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        # vLLM constructs TP groups first, and then construct other
        # parallel groups on top of TP groups.
        # for example, TP=4, PP=2,
        # PP group: [0, 1, 2, 3], [4, 5, 6, 7]
        # TP group: [0, 4], [1, 5], [2, 6], [3, 7]
        # So we can "exclude" the effect of TP by rank // tp_size.
        return world_size // tp_size, rank // tp_size


def create_scheduler_adapter(
    server_url: str,
    zmq_context: zmq.Context,
    vllm_config: VllmConfig,
    mq_timeout: float,
    heartbeat_interval: float,
) -> LMCacheMPSchedulerAdapter:
    world_size, kv_rank = extract_world_size_and_kv_rank(
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config,
    )
    parallel_strategy = ParallelStrategy(
        mla_enabled(vllm_config.model_config),
        world_size,
        kv_rank,
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config.parallel_config.tensor_parallel_size,
        vllm_config.parallel_config.pipeline_parallel_size,
    )

    return LMCacheMPSchedulerAdapter(
        server_url=server_url,
        context=zmq_context,
        model_name=vllm_config.model_config.model,
        vllm_block_size=vllm_config.cache_config.block_size,
        parallel_strategy=parallel_strategy,
        mq_timeout=mq_timeout,
        heartbeat_interval=heartbeat_interval,
    )


def create_worker_adapter(
    server_url: str,
    zmq_context: zmq.Context,
    vllm_config: VllmConfig,
    mq_timeout: float,
    heartbeat_interval: float,
) -> LMCacheMPWorkerAdapter:
    world_size, kv_rank = extract_world_size_and_kv_rank(
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config,
    )
    parallel_strategy = ParallelStrategy(
        mla_enabled(vllm_config.model_config),
        world_size,
        kv_rank,
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config.parallel_config.tensor_parallel_size,
        vllm_config.parallel_config.pipeline_parallel_size,
    )

    return LMCacheMPWorkerAdapter(
        server_url=server_url,
        context=zmq_context,
        model_name=vllm_config.model_config.model,
        vllm_block_size=vllm_config.cache_config.block_size,
        parallel_strategy=parallel_strategy,
        mq_timeout=mq_timeout,
        heartbeat_interval=heartbeat_interval,
    )


class LMCacheMPRequestState(enum.Enum):
    """
    State machine:
    PREFETCHING -- update_state_after_alloc --> WAITING_FOR_LOAD
    WAITING_FOR_LOAD -- process_loading_requests --> READY
    """

    PREFETCHING = enum.auto()
    WAITING_FOR_LOAD = enum.auto()
    READY = enum.auto()


@dataclass
class LMCacheMPRequestTracker:
    # NOTE: this class used vLLM data structures, should be part of
    # vLLM integration code

    request_id: str

    # Read-only lists to track the token ids and block hashes
    all_token_ids: ConstantList[int]
    block_hashes: ConstantList["BlockHash"]

    # Block ids and hashes will be updated at update_states_after_alloc and
    # during the generation
    allocated_block_ids: list[int] = field(default_factory=list)

    # Number of scheduled tokens in this request. We keep tracking this to
    # avoid saving half-full blocks.
    num_scheduled_tokens: int = 0

    # Number of blocks stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_blocks: int = 0

    # Staging load operation -- save vllm and lmcache hit tokens during lookup
    num_vllm_hit_blocks: int = 0
    num_lmcache_hit_blocks: int = 0

    # Main state
    state: LMCacheMPRequestState = LMCacheMPRequestState.PREFETCHING

    cache_salt: str = ""

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.cache_salt: str = request.cache_salt or ""
        self.all_token_ids = request.all_token_ids
        self.block_hashes = ConstantList(request.block_hashes)
        self.allocated_block_ids = []
        self.num_stored_blocks = 0
        self.num_vllm_hit_blocks = 0
        self.num_lmcache_hit_blocks = 0
        self.state = LMCacheMPRequestState.PREFETCHING

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_lmcache_hit_blocks > self.num_vllm_hit_blocks
            and self.state != LMCacheMPRequestState.READY
        )

    def is_ready_for_retrieving(self) -> bool:
        """Check whether the current request is ready for retrieving,
        will be used in process_loading_requests"""
        return (
            self.state == LMCacheMPRequestState.WAITING_FOR_LOAD
            and self.needs_retrieve()
        )

    ####
    # Update internal states
    ####
    def increase_num_scheduled_tokens(self, num_new_tokens: int):
        self.num_scheduled_tokens += num_new_tokens

    def increase_num_stored_blocks(self, num_new_blocks: int):
        """Increase the number of stored blocks for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_blocks += num_new_blocks

    def append_block_ids(
        self,
        new_block_ids: list[int],
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        self.allocated_block_ids.extend(new_block_ids)

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_block_hashes={len(self.block_hashes)}, "
            f"num_allocated_blocks={len(self.allocated_block_ids)}, "
            f"num_stored_blocks={self.num_stored_blocks}, "
            f"vllm_hit_blocks={self.num_vllm_hit_blocks}, "
            f"lmcache_hit_blocks={self.num_lmcache_hit_blocks}, "
            f"state={self.state})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class LMCacheMPRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp
    cache_salt: str = ""

    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
            vllm_block_size: the block size used in vLLM
        """
        # Store the blocks that has block hashes
        # NOTE: the invariant here is that `num_stored_blocks` should
        # always be a multiple of `blocks_in_chunk`
        # TODO: This should be checked everytime we update the num_stored_blocks
        #
        # Why computed_blocks uses max(num_vllm_hit_blocks, num_lmcache_hit_blocks):
        #
        # Both values represent a prefix of blocks whose KV data is already
        # available (either from vLLM APC or from LMCache), so they must NOT
        # be summed (that would double-count the overlapping prefix).
        #
        # * num_lmcache_hit_blocks: LMCache-hit blocks are already counted in
        #   num_stored_blocks (set during lookup), so they must be included
        #   here to keep the upper bound consistent.  They are NOT re-stored.
        # * num_vllm_hit_blocks: LMCache stores in units of chunks (N blocks),
        #   so num_lmcache_hit_blocks is rounded DOWN to the nearest chunk
        #   boundary.  When vLLM APC hits more blocks than that rounded value
        #   (e.g. APC=44 blocks, LMCache=32 blocks after chunk alignment),
        #   using only num_lmcache_hit_blocks would set the upper bound too
        #   low and silently skip the APC-hit blocks that fall between the
        #   two values, causing under-storing.  Taking the max ensures we
        #   always use the tighter (larger) of the two hit counts.
        computed_blocks = tracker.num_scheduled_tokens // vllm_block_size + max(
            tracker.num_vllm_hit_blocks, tracker.num_lmcache_hit_blocks
        )
        min_available_blocks = min(
            len(tracker.block_hashes),
            len(tracker.allocated_block_ids),
            computed_blocks,
        )
        num_staging_blocks = min_available_blocks - tracker.num_stored_blocks
        num_chunks = num_staging_blocks // blocks_in_chunk

        if num_chunks >= 1:
            start = tracker.num_stored_blocks
            end = start + num_chunks * blocks_in_chunk
            block_ids = tracker.allocated_block_ids[start:end]
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)
            block_hashes = list(tracker.block_hashes)[start:end]
            op = LoadStoreOp(
                token_ids=token_ids,
                block_hashes=block_hashes,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=op,
                cache_salt=tracker.cache_salt,
            )

            # Update the request tracker
            tracker.increase_num_stored_blocks(end - start)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
            vllm_block_size: the block size used in vLLM
        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_blocks |
        # | lmcache chunk 1   | lmcache chunk 2   |
        #                     |  need to retrieve |

        start = tracker.num_vllm_hit_blocks // blocks_in_chunk * blocks_in_chunk
        end = tracker.num_lmcache_hit_blocks
        assert end % blocks_in_chunk == 0, (
            "The number of LMCache hit blocks should be a multiple of the "
            "number of blocks in a lmcache chunk. "
        )
        assert len(tracker.block_hashes) >= end, (
            "The number of block hashes should be greater than or equal to the "
            "number of LMCache hit blocks. "
        )
        if end > start:
            block_ids = tracker.allocated_block_ids[start:end]
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)
            block_hashes = list(tracker.block_hashes)[start:end]

            # Compute how many tokens at the start of the retrieve range
            # overlap with APC-shared blocks. The server must skip writing
            # to these positions to avoid a cross-stream data race: the
            # retrieve writes on the LMCache CUDA stream while concurrent
            # requests may read these APC-shared blocks on the vLLM stream.
            apc_overlap_blocks = tracker.num_vllm_hit_blocks - start
            skip_first_n_tokens = apc_overlap_blocks * vllm_block_size

            op = LoadStoreOp(
                token_ids=token_ids,
                block_hashes=block_hashes,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
                skip_first_n_tokens=skip_first_n_tokens,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=op,
                cache_salt=tracker.cache_salt,
            )
            return ret

        return None


class LMCacheMPConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.requests: list[LMCacheMPRequestMetadata] = []

    def add_request_metadata(self, request_metadata: LMCacheMPRequestMetadata):
        self.requests.append(request_metadata)

    def __len__(self):
        return len(self.requests)

    # For debugging
    def __str__(self):
        request_strs = []
        for req_meta in self.requests:
            request_strs.append(
                f"RequestMetadata(request_id={req_meta.request_id}, "
                f"direction={req_meta.direction}, "
                f"num_blocks={len(req_meta.op)}, "
                f"block_ids={req_meta.op.block_ids})"
            )
        return "[" + "\n".join(request_strs) + "]"

    def __repr__(self):
        return self.__str__()


class LMCacheMPConnectorUpstream(KVConnectorBase_V1):
    """
    The connector for LMCache multi-process mode.

    Extra configs (kv_transfer_config.extra_config):
    - lmcache.mp.host: the host of the LMCache server.
    - lmcache.mp.port: the port of the LMCache server.
    - lmcache.mp.mq_timeout: timeout (seconds) for message queue requests.
    - lmcache.mp.heartbeat_interval: interval (seconds) between server
      heartbeat pings.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        server_host = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.host", "tcp://localhost"
        )
        server_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.port", 5555
        )
        mq_timeout = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.mq_timeout", 300.0
            )
        )
        heartbeat_interval = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.heartbeat_interval", 10.0
            )
        )

        server_url = f"{server_host}:{server_port}"
        zmq_context = zmq.Context.instance()
        if self.role == KVConnectorRole.SCHEDULER:
            self.scheduler_adapter = create_scheduler_adapter(
                server_url,
                zmq_context,
                vllm_config,
                mq_timeout,
                heartbeat_interval,
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
        elif self.role == KVConnectorRole.WORKER:
            self.worker_adapter = create_worker_adapter(
                server_url,
                zmq_context,
                vllm_config,
                mq_timeout,
                heartbeat_interval,
            )
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")

        self.vllm_block_size = vllm_config.cache_config.block_size

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """
        获取 connector 元数据。

        该方法应该只在 connector 内部调用。
        它返回当前步骤的 connector 元数据（由 build_connector_meta 构建）。

        注意：
        - 该方法仅在 connector 内部使用
        - 调用前必须确保 _connector_metadata 已经被设置

        Returns:
            KVConnectorMetadata: connector 元数据
        """

        # 断言：调用前必须已经设置了有效的元数据
        # _connector_metadata 在 build_connector_meta 被调用时设置
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        logger.info("Registering kv caches!")
        self.worker_adapter.register_kv_caches(kv_caches)
        return

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        从 LMCache 加载 KV cache 到 vLLM 的 paged KV buffer。

        该方法在 forward context 中、模型执行前被调用，用于：
        1. 从 connector 元数据中筛选 RETRIEVE 类型的请求
        2. 记录 CUDA 事件以确保异步加载的正确同步
        3. 提交批量的 retrieve 请求到 LMCache

        异步加载机制：
        - 使用 CUDA Event 实现跨进程同步
        - 模型执行时 KV 加载在后台进行
        - 通过 wait_for_layer_load 等待加载完成

        Args:
            forward_context: 前向传播上下文
            **kwargs: 加载操作的额外参数

        Note:
            kv_caches 和 layer_names 的元素数量应该相同。
        """
        # Step 1: 获取 connector 元数据
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        # Step 2: 初始化列表以收集 RETRIEVE 请求
        request_ids = []
        ops = []
        cache_salts = []

        # Step 3: 遍历所有请求元数据，筛选 RETRIEVE 类型的请求
        #   - direction: "RETRIEVE" 表示从 LMCache 加载 KV
        #   - direction: "STORE" 表示将 KV 存储到 LMCache（不处理）
        for meta in metadata.requests:
            if meta.direction != "RETRIEVE":
                continue
            # 收集该请求的相关信息
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)

        # Step 4: 如果没有 retrieve 请求，直接返回
        if len(request_ids) == 0:
            return

        # Step 5: 记录 CUDA 事件（用于跨进程同步）
        #   - event.interprocess=True: 允许跨进程共享事件
        #   - event.record(): 在当前流上记录事件
        with torch.cuda.stream(torch.cuda.current_stream()):
            event = torch.cuda.Event(interprocess=True)
            event.record()

        # Step 6: 提交批量的 retrieve 请求到 LMCache
        #   - LMCache Server 会在事件完成时执行实际的 KV 加载
        #   - 这是异步操作，不会阻塞模型执行
        self.worker_adapter.batched_submit_retrieve_requests(
            request_ids, ops, event, cache_salts=cache_salts
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        return

    def wait_for_save(self):
        """
        阻塞直到所有保存（store）操作完成。

        该方法在 forward context 退出时调用，确保 save_kv_layer
        启动的异步保存操作在 forward 结束前完成。

        这可以防止在保存完成前 paged KV buffer 被覆写。

        处理流程：
        1. MLA 场景检查（只让 PP group 的第一 rank 执行保存）
        2. 获取 connector 元数据
        3. 筛选 STORE 类型的请求
        4. 记录 CUDA 事件
        5. 提交批量的 store 请求

        注意：
        - 该方法只在 forward 结束时调用一次
        - STORE 请求是从 vLLM 到 LMCache 的写入操作
        - 与 start_load_kv 类似，使用 CUDA Event 跨进程同步
        """
        # Step 1: MLA 场景检查
        #   - 在 MLA 场景下，只有 PP group 的第一个 rank 需要保存 KV
        #   - 其他 rank 已经有第一 rank 处理的副本，不需要重复保存
        if (
            self.worker_adapter.use_mla
            and not self.worker_adapter.is_first_rank_of_pp_group
        ):
            return

        # Step 2: 获取 connector 元数据
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        # Step 3: 初始化列表以收集 STORE 请求
        request_ids = []
        ops = []
        cache_salts = []

        # Step 4: 遍历所有请求元数据，筛选 STORE 类型的请求
        #   - direction: "STORE" 表示将 KV 存储到 LMCache
        #   - direction: "RETRIEVE" 表示从 LMCache 加载（不处理）
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            # 收集该请求的相关信息
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)

        # Step 5: 如果没有 store 请求，直接返回
        if len(request_ids) == 0:
            return
            

        # Step 6: 记录 CUDA 事件（用于跨进程同步）
        #   - 与 start_load_kv 中的事件机制相同
        #   - event.interprocess=True: 允许跨进程共享事件
        with torch.cuda.stream(torch.cuda.current_stream()):
            event = torch.cuda.Event(interprocess=True)
            event.record()

        # Step 7: 提交批量的 store 请求到 LMCache
        #   - LMCache Server 会在事件完成时执行实际的 KV 存储
        #   - 这是异步操作，Worker 端不阻塞等待
        self.worker_adapter.batched_submit_store_requests(
            request_ids, ops, event, cache_salts=cache_salts
        )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        检查已完成异步传输的请求 ID。

        该方法被 Scheduler 进程调用，用于：
        1. 通知 Worker 端 connector 哪些请求已完成 token 生成
        2. Worker 端检查这些请求的异步传输（store/retrieve）是否完成
        3. 返回已完成的 store 和 retrieve 请求 ID

        Scheduler 进程（通过 Executors）会使用这个输出来跟踪
        哪些 Worker 已经完成。

        处理流程：
        1. 调用 worker_adapter.get_finished() 检查完成的请求
        2. 返回两元组：已完成的 store 请求 ID 和 retrieve 请求 ID

        注意：
        - 返回的 finished store/send 请求 ID 必须属于此方法调用
          （或之前调用）中提供的集合
        - 这是异步传输完成的回调机制

        Args:
            finished_req_ids: 引擎报告已完成的请求 ID 集合

        Returns:
            tuple[set[str] | None, set[str] | None]:
                - 第一个 set: 已完成的 store/send 请求 ID
                - 第二个 set: 已完成的 retrieve/load 请求 ID
        """
        # Step 1: 调用 Worker 端 adapter 检查已完成的请求
        #   - worker_adapter 会检查 retrieve_futures 和 store_futures
        #   - 找出异步传输已完成的请求
        val = self.worker_adapter.get_finished(finished_req_ids)

        # Step 2: 返回结果
        #   - val[0]: 已完成的 store 请求 ID 集合
        #   - val[1]: 已完成的 retrieve 请求 ID 集合
        return val

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        return self.worker_adapter.get_block_ids_with_load_errors()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        if hasattr(self, "worker_adapter"):
            self.worker_adapter.shutdown()
        return None

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        """
        Get the KV connector stats collected during the last interval.
        """
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        查询 LMCache 缓存中是否有当前请求的 KV cache，并返回可以加载的 token 数量。

        这个方法在请求调度前被调用，用于：
        1. 向 LMCache Server 提交 lookup 请求，检查是否有缓存的 KV
        2. 返回从外部缓存可以加载的 token 数量（相对于本地已计算的 token）
        3. 指示是否需要异步加载

        Args:
            request (Request): 请求对象，包含完整的 token 列表
            num_computed_tokens (int): 本地已计算的 token 数量（从之前的 prefill/decode 步骤）

        Returns:
            tuple[int | None, bool]: 返回元组
                - 第一个元素:
                  * int: 可以从外部缓存加载的 token 数量（超过本地已计算的部分）
                  * None: LMCache Server 还未返回结果，需要 scheduler 稍后再查询
                - 第二个元素:
                  * True: 外部 KV 将在调度步骤之间异步加载
                  * False: 同步加载（或不需要加载）

        Example:
            假设请求 prompt 为 [A,B,C,D,E,F,G,H] (8个token)，num_computed_tokens=2 (本地已计算前2个)
            - 如果 LMCache 缓存了全部 8 个 token: return (6, True)  # 还需加载6个新token (第3-8个)
            - 如果 LMCache 缓存了 0 个 token: return (0, False)    # 无需加载
            - 如果 LMCache 还未查询完: return (None, True)          # 需要稍后再查
        """
        # Step 1: 获取或创建请求追踪器，用于跟踪该请求的 KV 缓存状态
        tracker = self._get_or_create_request_tracker(request)

        # Step 2: 抢占的请求暂不支持从 LMCache 加载，直接返回 0
        # TODO: support loading KV for preempted requests in the future
        if request.status == RequestStatus.PREEMPTED:
            return 0, False

        # Step 3: 向 LMCache Server 提交 lookup 请求
        #   - 提交 block_hashes 用于 hash 模式匹配缓存中的 KV cache
        #   - cache_salt 用于区分相同 prompt 的不同请求（如不同采样参数）
        self.scheduler_adapter.maybe_submit_lookup_request(
            request.request_id,
            block_hashes=list(tracker.block_hashes),
            cache_salt=tracker.cache_salt,
        )

        # Step 4: 检查 lookup 结果
        #   - 返回命中的 token 数量（以字节为单位，需要转换）
        ret = self.scheduler_adapter.check_lookup_result(request.request_id)

        # 情况 A: Server 还未返回结果（异步查询中），需要稍后再试
        if ret is None:
            return None, True

        # 情况 B: 没有命中缓存，返回 0
        if ret == 0:
            return 0, False

        # Step 5: 验证返回值的对齐（确保 token 数与 block 大小对齐）
        assert (
            ret % (self.scheduler_adapter.num_blocks_per_chunk() * self.vllm_block_size)
            == 0
        )

        # Step 6: 计算 vLLM 本地已计算的 block 数 和 LMCache 命中的 block 数
        #   - num_vllm_blocks: 本地已计算的 block 数
        #   - num_lmcache_blocks: LMCache 命中的 block 数
        num_vllm_blocks = num_computed_tokens // self.vllm_block_size
        num_lmcache_blocks = ret // self.vllm_block_size
        tracker.increase_num_stored_blocks(num_lmcache_blocks)

        # Step 7: 保存命中信息到 tracker，供后续流程使用
        #   - num_vllm_hit_blocks: 本地已有的 block 数
        #   - num_lmcache_hit_blocks: LMCache 命中的 block 数
        tracker.num_vllm_hit_blocks = num_vllm_blocks
        tracker.num_lmcache_hit_blocks = num_lmcache_blocks

        # Step 8: 计算需要加载的 token 数量
        #   - need_to_load = LMCache命中 - 本地已计算
        #   - 这里的 ret 是 LMCache 命中的总 token 数（字节），需要与 num_computed_tokens 比较
        need_to_load = max(0, ret - num_computed_tokens)

        logger.debug(
            "vLLM hit is: %d, Need to load is %d", num_computed_tokens, need_to_load
        )

        # Step 9: 返回需要加载的 token 数和是否异步加载
        #   - need_to_load > 0 表示需要异步加载
        return need_to_load, need_to_load > 0

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        在 vLLM 为请求分配 KV cache blocks 后，更新 LMCache 的状态。

        该方法在以下场景被调用：
        1. 请求初次调度时，为其分配 KV cache blocks
        2. 异步加载完成后，可能需要分配额外的 blocks

        注意：对于需要从 LMCache 异步加载的请求，该方法可能被调用两次：
        - 第一次：分配初始 blocks (APC + 新 blocks)，用于加载外部 KV
        - 第二次：分配额外 blocks，用于剩余 token

        Args:
            request: 请求对象
            blocks: 分配给该请求的所有 KV cache blocks（注意：包含所有 blocks，不是只有新分配的）
            num_external_tokens: 将从外部 KV cache 加载的 token 数量

        Example:
            # 场景 1: 请求 "Hello World" (8 tokens)，LMCache 缓存了 4 个
            # - 第一次调用: blocks = [0,1] (2 blocks for 4 tokens)
            # - 第二次调用: blocks = [0,1,2,3] (4 blocks for 8 tokens)
            #
            # 场景 2: 本地已计算 2 tokens，LMCache 缓存了 6 tokens
            # - 需要加载 4 tokens (6 - 2)
            # - 分配 1 个 block (4 tokens)
        """
        # 注意：blocks 来自 kv_cache_manager.get_blocks(request_id)
        # 它返回请求的 ALL blocks（不是只有新分配的）
        # 对于异步加载的请求，这个方法可能被调用两次：
        #   第一次调用: blocks = 初始分配 (APC + 新 blocks)
        #   第二次调用: blocks = 所有 blocks (初始 + 新分配的)
        # 我们只能追加新分配的 blocks，避免重复导致 store 路径的 block 索引混乱

        # Step 1: 获取请求追踪器
        tracker = self._get_request_tracker(request.request_id)

        # Step 2: 获取分配的 block IDs 并格式化
        block_ids = reformat_block_ids(blocks.get_block_ids())

        # Step 3: 只追加尚未追踪的新 blocks
        #   - existing_count: 已有多少 blocks 被追踪
        #   - new_block_ids: 新分配的 blocks
        existing_count = len(tracker.allocated_block_ids)
        new_block_ids = block_ids[existing_count:]
        if new_block_ids:
            tracker.append_block_ids(new_block_ids)

        # Step 4: 更新 tracker 的状态
        #   - 检查是否需要从 LMCache retrieve
        condition = tracker.needs_retrieve()

        if tracker.state == LMCacheMPRequestState.PREFETCHING:
            # 状态转换：PREFETCHING → WAITING_FOR_LOAD 或 READY
            #   - 需要 retrieve: → WAITING_FOR_LOAD (等待加载)
            #   - 不需要 retrieve: → READY (准备好执行)
            tracker.state = (
                LMCacheMPRequestState.WAITING_FOR_LOAD
                if condition
                else LMCacheMPRequestState.READY
            )

            # 清理 lookup future，防止内存泄漏
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)

            # Step 5: 释放已被 vLLM 本地计算、不需要从 LMCache retrieve 的 chunks 的锁
            #   - LMCache 在 lookup 时会锁定 chunks，防止被驱逐
            #   - 如果某些 token 已经在 vLLM 本地计算过了，需要释放对应的锁
            if tracker.num_lmcache_hit_blocks > 0:
                if not condition:
                    # 不需要 retrieve：释放所有锁定的 chunks
                    free_end = tracker.num_lmcache_hit_blocks * self.vllm_block_size
                else:
                    # 需要 retrieve：只释放 vLLM 已计算的 token 范围
                    # 注意：vLLM blocks 和 LMCache blocks 边界可能不对齐
                    # free_lookup_locks 会处理这种边界不对齐情况
                    free_end = tracker.num_vllm_hit_blocks * self.vllm_block_size

                if free_end > 0:
                    # 通知 LMCache Server 释放指定范围的锁
                    # Hash 模式下直接传递 block_hashes
                    free_end_blocks = (
                        tracker.num_lmcache_hit_blocks
                        if not condition
                        else tracker.num_vllm_hit_blocks
                    )
                    self.scheduler_adapter.free_lookup_locks(
                        block_hashes=list(tracker.block_hashes)[:free_end_blocks],
                        request_id=request.request_id,
                    )
                    logger.debug(
                        "Free locks of tokens %d-%d since it is cached by vLLM.",
                        0,
                        free_end,
                    )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        构建当前调度步骤的 KVConnector 元数据。

        该方法在每个调度周期被调用，负责：
        1. 处理需要从 LMCache retrieve 的请求
        2. 处理新请求的存储 (store) 元数据
        3. 处理缓存请求的存储 (store) 元数据
        4. 报告 block 分配变化以便观测

        注意：
        - 此方法不应该修改 scheduler_output 中的任何字段
        - 调用此方法会重置 connector 的状态

        Args:
            scheduler_output: 调度器输出对象，包含本轮调度的请求信息

        Returns:
            KVConnectorMetadata: 包含所有需要执行的 KV 传输操作的元数据
        """
        # Step 1: 创建元数据容器
        metadata = LMCacheMPConnectorMetadata()

        # Step 2: 处理需要从 LMCache retrieve 的请求（异步加载完成的请求）
        self._process_retrieve_requests(metadata)

        # Step 3: 处理新请求（首次调度的请求）的存储元数据
        self._process_new_requests(scheduler_output, metadata)

        # Step 4: 处理缓存请求（从检查点恢复的请求）的存储元数据
        self._process_cached_requests(scheduler_output, metadata)

        # Step 5: 记录日志
        if len(metadata) > 0:
            logger.debug("Final connector metadata: %s", metadata)

        # Step 6: 向 LMCache 报告 block 分配变化，用于可观测性
        self._report_block_allocation_deltas(scheduler_output)

        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        当请求完成时调用（在 blocks 被释放之前）。

        该方法执行清理工作：
        1. 计算额外的缓存 token 数量
        2. 清理请求追踪器以防止内存泄漏
        3. 通知 LMCache 结束该请求的 session

        返回值说明：
        - True: 请求正在异步保存/发送，blocks 不应该被释放，
                直到 request_id 从 get_finished() 返回
        - return_params: 包含在请求输出中的可选 KVTransferParams

        Args:
            request: 请求对象
            block_ids: 请求使用的 block IDs

        Returns:
            tuple[bool, dict | None]:
                - bool: 是否异步处理（始终返回 True）
                - dict | None: 返回参数（包含额外缓存的 token 数量）
        """
        # Step 1: 获取请求的 KV 传输参数
        params: dict[str, Any] | None = getattr(request, "kv_transfer_params", None)
        return_params: dict[str, Any] | None = {} if params is not None else None

        # Step 2: 计算额外的缓存 token 数量
        # 如果请求有 kv_transfer_params 且包含 num_lmcache_extra_cached_tokens，
        # 说明有 LMCache 额外缓存的 token，需要返回这个数量
        if (
            params is not None
            and return_params is not None
            and "num_lmcache_extra_cached_tokens" in params
        ):
            request_tracker = self._get_request_tracker(request.request_id)
            # 计算额外的缓存 blocks（LMCache 命中但 vLLM 未命中的部分）
            num_extra_cached_blocks = max(
                0,
                request_tracker.num_lmcache_hit_blocks
                - request_tracker.num_vllm_hit_blocks,
            )
            # 转换为 token 数量
            return_params["num_lmcache_extra_cached_tokens"] = (
                num_extra_cached_blocks * self.vllm_block_size
            )

        # Step 3: 清理请求追踪器，防止内存泄漏
        self._cleanup_request_tracker(request.request_id)

        # Step 4: 通知 LMCache Server 结束该请求的 session
        self.scheduler_adapter.end_session(request.request_id)

        # Step 5: 返回 True 表示请求正在异步保存，blocks 暂不释放
        return True, return_params

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        return ()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        return None

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats | None":
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> "KVConnectorPromMetrics | None":
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        return None

    ##############################
    # Helper functions
    ##############################
    def _process_retrieve_requests(
        self,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        """
        处理需要从 LMCache retrieve（加载）的请求。

        当请求状态为 WAITING_FOR_LOAD 时，表示异步加载已完成，
        需要构建 retrieve 元数据以便 worker 执行 KV 传输。

        处理流程：
        1. 遍历所有请求追踪器
        2. 筛选出状态为 WAITING_FOR_LOAD 的请求
        3. 为每个请求构建 retrieve 元数据
        4. 将元数据添加到 connector metadata
        5. 将请求状态更新为 READY

        Args:
            metadata: LMCacheMPConnectorMetadata 对象，用于收集 retrieve 请求元数据
        """
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        # 遍历所有请求追踪器
        for request_tracker in self.request_trackers.values():
            # 只处理状态为 WAITING_FOR_LOAD 的请求
            if request_tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue

            # 构建该请求的 retrieve 元数据
            r_metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
                request_tracker,
                blocks_per_chunk,
                vllm_block_size=self.vllm_block_size,
            )

            # 添加到元数据集合
            if r_metadata is not None:
                metadata.add_request_metadata(r_metadata)

            # 状态转换为 READY（准备好执行）
            request_tracker.state = LMCacheMPRequestState.READY

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        """
        处理新请求（首次调度的请求）的存储元数据。

        新请求是指首次被调度的请求，需要将其 KV cache 存储到 LMCache。
        该方法为每个新请求构建 store 元数据，以便 worker 执行 KV 存储操作。

        处理流程：
        1. 遍历 scheduler_output 中的新请求列表
        2. 获取对应请求的追踪器
        3. 更新已调度 token 计数
        4. 构建 store 元数据
        5. 添加到 connector metadata

        Args:
            scheduler_output: 调度器输出对象，包含新请求列表
            metadata: LMCacheMPConnectorMetadata 对象，用于收集 store 请求元数据
        """
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        # 遍历所有新请求（首次调度的请求）
        for new_request in scheduler_output.scheduled_new_reqs:
            # 获取该请求的追踪器
            request_tracker = self._get_request_tracker(new_request.req_id)

            # 更新已调度的 token 数量
            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            # 构建 store 元数据（用于将 KV 存储到 LMCache）
            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker, blocks_per_chunk, self.vllm_block_size
            )

            # 添加到元数据集合
            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        """
        处理缓存请求（从检查点恢复的请求）的存储元数据。

        缓存请求是指之前已经部分执行过的请求，现在从检查点恢复继续执行。
        该方法为每个缓存请求构建 store 元数据，以便 worker 执行 KV 存储操作。

        与新请求的区别：
        - 缓存请求可能已经有部分 KV 在本地
        - 需要追加新分配的 blocks
        - 可能需要处理恢复（resume）的请求

        处理流程：
        1. 遍历 scheduler_output 中的缓存请求列表
        2. 获取对应请求的追踪器
        3. 追加新分配的 blocks（非恢复请求）
        4. 更新已调度 token 计数
        5. 构建 store 元数据
        6. 添加到 connector metadata

        Args:
            scheduler_output: 调度器输出对象，包含缓存请求列表
            metadata: LMCacheMPConnectorMetadata 对象，用于收集 store 请求元数据
        """
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        # 获取缓存请求列表
        cached_reqs = scheduler_output.scheduled_cached_reqs

        # 遍历所有缓存请求（从检查点恢复的请求）
        for idx, request_id in enumerate(cached_reqs.req_ids):
            # 获取该请求的追踪器
            request_tracker = self._get_request_tracker(request_id)

            # 追加新分配的 blocks
            new_block_ids = reformat_block_ids(cached_reqs.new_block_ids[idx])
            # 注意：恢复的请求不追加新 blocks（因为它们已有完整的 blocks）
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids(new_block_ids)

            # 更新已调度的 token 数量（与 _process_new_requests 保持一致）
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            # 构建 store 元数据
            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker, blocks_per_chunk, self.vllm_block_size
            )

            # 添加到元数据集合
            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _report_block_allocation_deltas(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """
        收集每个请求的 block 分配变化并报告给 LMCache。

        该方法用于可观测性目的，帮助 LMCache 了解每个请求的 block 分配情况：
        - 新请求：报告所有已分配的 block_ids 和 token_ids
        - 缓存请求：只报告新追加的 block_ids 和 token_ids

        用途：
        - 用于 L0 指标订阅，正确地将每个 block 映射到其实际的 token 内容
        - 用于监控和调试 block 分配情况

        Args:
            scheduler_output: 调度器输出对象
        """
        records: list[RequestAllocationRecord] = []

        # New requests: send all tokens covering all allocated blocks so
        # the L0 metrics subscriber can correctly map each block to its
        # actual token content (not just the newly-scheduled slice).
        for new_request in scheduler_output.scheduled_new_reqs:
            tracker = self.request_trackers.get(new_request.req_id)
            if tracker is None:
                continue
            num_blocks = len(tracker.allocated_block_ids)
            total_tokens = num_blocks * self.vllm_block_size
            records.append(
                RequestAllocationRecord(
                    req_id=new_request.req_id,
                    new_block_ids=list(tracker.allocated_block_ids),
                    new_token_ids=list(tracker.all_token_ids[:total_tokens]),
                )
            )

        # Cached requests: only the newly added blocks and their full
        # token content.  We send all tokens covered by the new blocks
        # (not just the tokens scheduled this step) so the L0 subscriber
        # can correctly identify block content.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            new_block_ids = reformat_block_ids(cached_reqs.new_block_ids[idx])
            if not new_block_ids:
                continue
            tracker = self.request_trackers.get(request_id)
            if tracker is None:
                continue
            # The new blocks sit at the end of the request's block list.
            # Compute the token range they cover.
            total_blocks = len(tracker.allocated_block_ids)
            num_new_blocks = len(new_block_ids)
            start_token = (total_blocks - num_new_blocks) * self.vllm_block_size
            end_token = total_blocks * self.vllm_block_size
            new_token_ids = list(tracker.all_token_ids[start_token:end_token])
            records.append(
                RequestAllocationRecord(
                    req_id=request_id,
                    new_block_ids=new_block_ids,
                    new_token_ids=new_token_ids,
                )
            )

        if records:
            self.scheduler_adapter.report_block_allocations(records)

    def _get_request_tracker(self, request_id: str) -> LMCacheMPRequestTracker:
        assert request_id in self.request_trackers, (
            f"Request tracker for request_id {request_id} not found. "
        )
        return self.request_trackers[request_id]

    def _get_or_create_request_tracker(
        self, request: "Request"
    ) -> LMCacheMPRequestTracker:
        request_id = request.request_id
        # Remove the old trackers that is created before the preemption
        if (
            request.status == RequestStatus.PREEMPTED
            and request_id in self.request_trackers
        ):
            tracker = self.request_trackers[request_id]

            # NOTE: since this function may be called multiple times
            # for a single request (because get_num_new_matched_tokens
            # may be called multiple times) for the same request, we
            # will only do the remove if the tracker is not in the "fresh"
            # state, i.e., PREFETCHING
            if tracker.state != LMCacheMPRequestState.PREFETCHING:
                self.request_trackers.pop(request_id)

        if request_id not in self.request_trackers:
            new_tracker = LMCacheMPRequestTracker(request)
            self.request_trackers[request_id] = new_tracker
        return self.request_trackers[request_id]

    def _cleanup_request_tracker(self, request_id: str) -> None:
        """
        清理请求追踪器及其相关的 lookup future，防止内存泄漏。

        当请求完成时调用，确保：
        1. 请求追踪器被移除
        2. 相关的资源被释放

        Args:
            request_id: 要清理的请求 ID
        """
        # 从 request_trackers 字典中移除该请求的追踪器
        if self.request_trackers.pop(request_id, None):
            logger.debug(
                "[KVConnector] Cleaned up request_tracker for request %s",
                request_id,
            )


# At module load time, prefer the external LMCacheMPConnector shipped with the
# ``lmcache`` package. This avoids forcing users to set
# ``kv_connector_module_path`` when they only configure ``kv_connector``. If
# the external module is unavailable (e.g. older lmcache version that does
# not ship this submodule, or any import error), fall back to the builtin
# implementation defined above.
def _resolve_lmcache_mp_connector() -> type[KVConnectorBase_V1]:
    if os.environ.get("LMCACHE_USE_UPSTREAM_MP"):
        logger.info(
            "Force use builtin LMCacheMPConnectorUpstream in vLLM.",
        )
        return LMCacheMPConnectorUpstream

    try:
        from lmcache.integration.vllm.lmcache_mp_connector import (
            LMCacheMPConnector as _ExternalLMCacheMPConnector,
        )

        logger.info(
            "Using external LMCacheMPConnector from "
            "lmcache.integration.vllm.lmcache_mp_connector"
        )
        return _ExternalLMCacheMPConnector
    except ImportError as e:
        logger.info(
            "External LMCacheMPConnector is not available (%s), "
            "falling back to builtin implementation in vLLM.",
            e,
        )
        return LMCacheMPConnectorUpstream


LMCacheMPConnector = _resolve_lmcache_mp_connector()
