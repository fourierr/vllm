# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import zmq
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

logger = init_logger(__name__)


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def striding_block_hashes(
    block_hashes: list[bytes], blocks_in_chunk: int
) -> Iterable[bytes]:
    """Extract chunk-level hashes from block hashes by striding.

    In hash-based vLLM, each vLLM block has its own hash.  LMCache chunks
    span ``blocks_in_chunk`` consecutive blocks.  The representative hash
    for a chunk is the hash of the **last** block in that chunk (because
    each block hash already encodes its prefix).  So we start at index
    ``blocks_in_chunk - 1`` and stride by ``blocks_in_chunk``.
    """
    return islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk)


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result()
    return chunk_size


@dataclass
class ParallelStrategy:
    use_mla: bool
    """Whether to use the MLA."""

    kv_world_size: int
    """
    The kv world size, kv_world_size may not be equal to the actual_world_size, 
    in the case of mla, it will 'exclude' the effect of TP, the value is 
    calculated by `extract_world_size_and_kv_rank` in `lmcache_mp_connector.py`.
    """

    kv_worker_id: int
    """
    The kv worker id of the sub-process, kv_worker_id may not be equal to the 
    actual_worker_id, in the case of mla, it will 'exclude' the effect of TP, 
    the value is calculated by `extract_world_size_and_kv_rank` in 
    `lmcache_mp_connector.py`.
    """

    actual_world_size: int
    """The actual world size."""

    actual_worker_id: int
    """The actual worker id of the sub-process."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""


@dataclass
class LoadStoreOp:
    block_ids: list[int]
    """Block ids for the load/store operation"""

    token_ids: list[int] | None = None
    """Token IDs for the load/store operation (token mode)"""

    block_hashes: list[bytes] | None = None
    """Block hashes for the load/store operation (hash mode)"""

    start: int = 0
    """Start token index (token mode only)"""

    end: int = 0
    """End token index (token mode only)"""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = list[bool]
LookupResult = int


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
    ):
        """
        Args:
            server_url: The server URL for the LMCache message queue
            context: The ZMQ context

            model_name: The model name used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes `use_mla`,
                `world_size`, `worker_id` and so on
        """
        self.mq_client = MessageQueueClient(server_url, context)

        # Request futures
        self.lookup_futures: dict[str, MessagingFuture[LookupResult]] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        self.chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert self.chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

    @property
    def world_size(self) -> int:
        """The world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """The worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def tp_size(self) -> int:
        """The tensor parallel size."""
        return self.parallel_strategy.tp_size

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        block_hashes: list[bytes] | None = None,
        token_ids: list[int] | None = None,
    ) -> None:
        """
        向 LMCache Server 提交 lookup 请求，查询是否有缓存的 KV cache。

        该方法支持两种模式：
        - token_ids 模式 (token-based vLLM): 使用 token ID 列表作为 key
        - block_hashes 模式 (hash-based vLLM): 使用 block hash 列表作为 key

        注意：只有当该请求没有正在进行的 lookup 请求时才会提交新请求。

        Args:
            request_id: 请求 ID，用于标识同一个请求的多次查询
            block_hashes: block hash 列表 (hash 模式)
            token_ids: token ID 列表 (token 模式)

        Returns:
            None

        注意 (Notes):
            这个方法有副作用：提交 lookup 请求后，会"锁定" LMCache 中的 KV cache chunks，
            以便后续的 retrieve 操作可以获取这些缓存。
            同时，该方法会记录 lookup 请求的状态，可以通过 check_lookup_result 查看结果。

        Example:
            # Token 模式示例
            maybe_submit_lookup_request(
                request_id="req_123",
                token_ids=[101, 102, 103, 104, 105]  # "Hello World"
            )

            # Hash 模式示例
            maybe_submit_lookup_request(
                request_id="req_456",
                block_hashes=[b'hash1', b'hash2', b'hash3']
            )
        """
        # Step 1: 检查是否已有进行中的 lookup 请求
        #   - 避免重复提交，提高效率
        if request_id in self.lookup_futures:
            # Skip if there is already a lookup request
            return

        # Step 2: 参数校验 - 只能二选一
        assert (block_hashes is None) != (token_ids is None), (
            "Exactly one of block_hashes or token_ids must be provided"
        )

        # Step 3: 根据模式构建缓存 key
        if block_hashes is not None:
            # Hash 模式: 对 block hashes 进行步长处理，生成 chunk 级别的 hash key
            #   - block_hashes: [h0, h1, h2, h3, h4, h5] (每个 block 一个 hash)
            #   - striding: 选取每 N 个 block 的最后一个 hash 作为 chunk 的 key
            #   - 结果: [h2, h5, ...] (假设 blocks_in_chunk=3)
            chunk_hashes = list(
                striding_block_hashes(block_hashes, self.blocks_in_chunk)
            )
            # 为每个 chunk hash 创建 key
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token 模式: 将 token 列表对齐到 chunk 边界
            #   - 只保留完整的 chunk，避免部分 chunk
            #   - 例如: [1,2,3,4,5,6], chunk_size=4 → 只保留 [1,2,3,4]
            assert token_ids is not None
            aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size
            # 如果没有完整的 chunk，直接返回
            if aligned_end == 0:
                return
            # 创建 token-mode key (不需要 worker_id 版本)
            keys = [
                self._create_key(
                    token_ids,
                    start=0,
                    end=aligned_end,
                    request_id=request_id,
                ).no_worker_id_version()
            ]

        # Step 4: 发送 LOOKUP 请求到 LMCache Server
        #   - 使用 ZMQ 异步发送
        #   - 请求类型为 RequestType.LOOKUP
        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [keys],
        )

        # Step 5: 保存 future 到字典，供后续 check_lookup_result 使用
        #   - 后续可以通过 request_id 查询结果
        self.lookup_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        检查之前提交的 lookup 请求的结果。

        该方法是异步非阻塞的：
        - 如果请求还未完成，返回 None，调用者需要稍后再试
        - 如果请求已完成，返回命中的 token 总数（前缀匹配）

        Args:
            request_id: 在 maybe_submit_lookup_request 中提交的请求 ID

        Returns:
            int | None:
                - int: LMCache 中匹配的前缀 token 总数
                - None: lookup 请求还未完成，需要稍后再查询

        Example:
            # 场景: 请求 "Hello World" 的 KV cache
            # 假设 LMCache 缓存了 "Hello" 的 KV，但没缓存 "World"
            # 返回: 5 (表示前5个token有缓存)
        """
        # Step 1: 断言请求已提交（防止查询未提交的请求）
        assert request_id in self.lookup_futures, (
            f"Lookup request for request_id={request_id} has not been submitted"
        )

        # Step 2: 获取之前保存的 future 对象
        future = self.lookup_futures[request_id]

        # Step 3: 非阻塞查询请求状态
        #   - future.query() 是非阻塞的，只检查是否完成
        #   - 如果未完成，返回 None，调度器会稍后再试
        if not future.query():
            return None

        # Step 4: 获取结果并转换
        #   - future.result() 获取实际的命中 chunk 数量
        #   - 将 chunk 数量转换为 token 数量
        result = future.result()
        num_chunks = result
        # 例如: 2 chunks * 4 tokens/chunk = 8 tokens
        return num_chunks * self.chunk_size

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup future for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self.lookup_futures.pop(request_id, None)

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        send_lmcache_request(
            self.mq_client,
            RequestType.END_SESSION,
            [request_id],
        )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str | None = None,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            tp_size=self.tp_size,
        )

    def _create_hash_key(
        self, chunk_hash: bytes, request_id: str | None = None
    ) -> IPCCacheEngineKey:
        """Create a hash-mode IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            chunk_hash=chunk_hash,
            request_id=request_id,
            tp_size=self.tp_size,
        )


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
    ):
        self.mq_client = MessageQueueClient(server_url, context)

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        # request_id -> (future, other merged requests)
        self.store_futures: dict[
            str, tuple[MessagingFuture[StoreResult], list[str]]
        ] = {}
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[str]]
        ] = {}

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size

    @property
    def world_size(self) -> int:
        """The world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """The worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def use_mla(self) -> bool:
        """Whether to use MLA."""
        return self.parallel_strategy.use_mla

    @property
    def is_first_rank_of_pp_group(self) -> bool:
        """Is the first rank of the pipeline parallel group."""
        return (
            self.parallel_strategy.actual_worker_id % self.parallel_strategy.tp_size
            == 0
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Register the kv caches with LMCache server

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
        """
        # Register kv cache and send the request
        self.kv_caches = kv_caches
        logger.info("Registering kv caches")
        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [self.instance_id, wrap_kv_caches(kv_caches)],
        )
        future.result()

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        if op.block_hashes is not None:
            # Hash mode
            chunk_hashes = list(
                striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode
            assert op.token_ids is not None
            keys = [
                self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
            ]
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_id] = (future, [])

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        if op.block_hashes is not None:
            # Hash mode
            chunk_hashes = list(
                striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode
            assert op.token_ids is not None
            keys = [
                self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
            ]
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, [])

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        批量提交 store 请求到 LMCache Server。

        该方法将多个请求的 KV cache 存储操作打包成一次
        跨进程消息发送到 LMCache Server，由 Server 端执行
        实际的 KV 存储。

        处理流程：
        1. 遍历所有 (request_id, op) 对
        2. 根据 op 类型构建 IPC cache engine keys（hash 模式或 token 模式）
        3. 收集 block_ids
        4. 通过 ZMQ 消息队列发送 store 请求
        5. 记录 Future 用于后续完成检查

        Args:
            request_ids: 请求 ID 列表
            ops: LoadStoreOp 列表，描述每个请求的 store 操作
            event: 当前模型推理步骤后记录的 CUDA Event

        Returns:
            None

        Notes:
            - 该方法是异步的，调用后立即返回
            - LMCache Server 端会等待 event 触发后才执行存储
            - 通过 store_futures 跟踪完成状态
        """
        # Step 1: 初始化 IPC keys 和 block_ids 列表
        all_keys: list[IPCCacheEngineKey] = []
        block_ids: list[int] = []

        # Step 2: 遍历每个请求，根据 op 类型构建 keys
        for request_id, op in zip(request_ids, ops, strict=False):
            if op.block_hashes is not None:
                # Hash 模式：使用 block_hashes 作为 key
                #   - 通过 striding_block_hashes 将连续的 block 转换为 LMCache chunk
                #   - 每个 chunk 对应一个 hash key
                chunk_hashes = list(
                    striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
                )
                # 为每个 chunk hash 创建一个 key
                keys = [
                    self._create_hash_key(ch, request_id=request_id)
                    for ch in chunk_hashes
                ]
                all_keys.extend(keys)
            else:
                # Token 模式：使用 token_ids 作为 key（默认模式）
                #   - 直接使用 token_ids 区间 [start, end) 作为 key
                #   - 注意：每个请求只生成一个 key（不是 chunk 级别的）
                assert op.token_ids is not None
                all_keys.append(
                    self._create_key(
                        op.token_ids, op.start, op.end, request_id=request_id
                    )
                )
            # 收集该请求的所有 block_ids
            block_ids.extend(op.block_ids)

        # Step 3: 发送 store 请求到 LMCache Server
        #   - 通过 ZMQ 消息队列发送
        #   - 消息内容:
        #     [all_keys, instance_id, block_ids, event.ipc_handle()]
        #   - event.ipc_handle() 用于跨进程同步，保证 Server 在
        #     vLLM Worker 的 CUDA 流执行到 event.record() 后才执行存储
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [
                all_keys,            # 要存储的所有 chunk keys
                self.instance_id,     # vLLM 实例 ID
                block_ids,            # vLLM 分配的 block IDs（数据来源）
                event.ipc_handle(),   # CUDA Event IPC 句柄
            ],
        ).to_cuda_future()

        # Step 4: 存储 Future 用于后续完成检查
        #   - 以 request_ids[0] 为主键（批量提交的代表性 ID）
        #   - other_reqs 记录其他请求 ID，完成时一起更新
        self.store_futures[request_ids[0]] = (future, list(request_ids[1:]))

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        批量提交 retrieve 请求到 LMCache Server。

        该方法从 LMCache 加载 KV cache 到 vLLM 的 paged KV buffer。
        典型的调用场景是在 start_load_kv 中，模型执行前触发。

        Args:
            request_ids: 需要 retrieve 的请求 ID 列表
            ops: LoadStoreOp 列表，描述每个请求的 retrieve 操作。
                长度应与 request_ids 相同
            event: CUDA Event，在当前模型推理步骤后记录。
                用于跨进程同步，确保 LMCache 在 vLLM 流执行到该点后才开始 retrieve

        处理流程:
        ┌─────────────────────────────────────────────────────────────────────┐
        │ Step 1: 构建 IPC keys                                             │
        │   - 遍历 (request_id, op) 对                                      │
        │   - 两种模式:                                                      │
        │     A) hash 模式: op.block_hashes is not None                     │
        │        - striding_block_hashes 将 block_hashes 转换为 chunk hashes│
        │        - _create_hash_key 创建 hash 形式的 key                    │
        │     B) token 模式: op.token_ids is not None (默认)                │
        │        - _create_key 创建 token 形式的 key                        │
        │   - 同时收集所有请求的 block_ids                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │ Step 2: 发送跨进程消息                                             │
        │   - send_lmcache_request(mq_client, RequestType.RETRIEVE, [...])│
        │   - 消息内容:                                                    │
        │     [all_keys, instance_id, block_ids, event.ipc_handle()]       │
        │       - all_keys: 要检索的 KV keys                                │
        │       - instance_id: vLLM 实例 ID，用于定位目标位置                │
        │       - block_ids: vLLM 分配的 block 位置，KV 加载到此处            │
        │       - event.ipc_handle(): CUDA Event IPC 句柄用于跨进程同步    │
        │   - 返回 Future                                                  │
        ├─────────────────────────────────────────────────────────────────────┤
        │ Step 3: Future 管理                                               │
        │   - to_cuda_future() 将 LMCache Future 转换为 CUDA Future        │
        │   - self.retrieve_futures[request_ids[0]] = (future, other_reqs)│
        │   - 用于后续通过 get_finished 检查请求完成状态                    │
        └─────────────────────────────────────────────────────────────────────┘

        数据流向:
            LMCache Server ──► KV 数据 ──► vLLM Worker
                                  │
                            通过 block_ids 写入
                            vLLM 的 paged KV buffer

        与 batched_submit_store_requests 对比:
            Store: vLLM ──► LMCache (数据流出)
            Retrieve: LMCache ──► vLLM (数据流入)
            两者的数据流向相反，但 key 构建和 Future 管理逻辑相同

        注意:
            - 这是异步操作，不会阻塞 vLLM 继续执行
            - LMCache Server 端会等待 event 触发后才开始实际的 retrieve
            - 跨进程同步由 CUDA Event IPC handle 保证
        """
        all_keys: list[IPCCacheEngineKey] = []
        block_ids: list[int] = []
        for request_id, op in zip(request_ids, ops, strict=False):
            if op.block_hashes is not None:
                # Hash 模式：使用 vLLM 已计算的 block_hashes
                # 将 striding block_hashes 转换为 chunk 级别的 hashes
                chunk_hashes = list(
                    striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
                )
                keys = [
                    self._create_hash_key(ch, request_id=request_id)
                    for ch in chunk_hashes
                ]
                all_keys.extend(keys)
            else:
                # Token 模式 (默认)：使用 token_ids 构建 key
                assert op.token_ids is not None
                all_keys.append(
                    self._create_key(
                        op.token_ids, op.start, op.end, request_id=request_id
                    )
                )
            # 收集 vLLM 分配的 block 位置（retrieve 目标位置）
            block_ids.extend(op.block_ids)

        # 发送 RETRIEVE 请求到 LMCache Server
        # LMCache Server 收到消息后会:
        # 1. 等待 event 触发（跨进程同步）
        # 2. 根据 all_keys 查找 LMCache 中的 KV 数据
        # 3. 通过 instance_id 和 block_ids 将数据写入 vLLM
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                all_keys,            # 要检索的 KV keys
                self.instance_id,     # vLLM 实例 ID（定位目标位置）
                block_ids,            # vLLM 分配的 block 位置（写入目标）
                event.ipc_handle(),   # CUDA Event IPC 句柄（同步）
            ],
        ).to_cuda_future()

        # 存储 Future 用于后续检查完成状态
        # request_ids[0] 作为主 key，其他请求 ID 存在 other_reqs 中
        # 完成时一起标记为已完成
        self.retrieve_futures[request_ids[0]] = (future, list(request_ids[1:]))

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        检查并获取已完成的 store 和 retrieve 请求。

        该方法被 Scheduler 端调用，用于：
        1. 检查哪些异步 store 请求已经完成
        2. 检查哪些异步 retrieve 请求已经完成
        3. 返回两组已完成的请求 ID

        处理流程：
        1. 遍历 store_futures，检查每个 Future 是否完成
        2. 遍历 retrieve_futures，检查每个 Future 是否完成
        3. 清理已完成的 Future
        4. 更新内部状态并返回结果

        Args:
            finished_req_ids_from_engine: 引擎报告已完成的请求 ID 集合

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        # Step 1: 初始化完成集合
        finished_stores = set()
        finished_retrieves = set()

        # Step 2: 检查所有 store Future 是否完成
        #   - store_futures: dict[request_id, (Future, other_reqs)]
        #   - query() 非阻塞检查 Future 是否完成
        #   - result() 阻塞获取 Future 结果
        for request_id, (s_future, other_reqs) in self.store_futures.items():
            # Future 未完成则跳过
            if not s_future.query():
                continue

            # Future 已完成，获取结果
            s_result = s_future.result()
            # 收集完成的 store 请求 ID
            finished_stores.add(request_id)
            finished_stores.update(other_reqs)

            # 错误处理
            if not s_result:
                # TODO: add error handling here
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        # Step 3: 检查所有 retrieve Future 是否完成
        for request_id, (r_future, other_reqs) in self.retrieve_futures.items():
            # Future 未完成则跳过
            if not r_future.query():
                continue

            # Future 已完成，获取结果
            r_result = r_future.result()
            # 收集完成的 retrieve 请求 ID
            finished_retrieves.add(request_id)
            finished_retrieves.update(other_reqs)

            # 错误处理
            if not all(r_result):
                # TODO: add error handing here
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Step 4: 从跟踪字典中移除已完成的请求
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

        # Step 5: 更新内部状态
        #   - 将已完成 store 请求添加到 finished_stores 集合
        self.finished_stores.update(finished_stores)

        # Step 6: 交叉引用引擎报告的完成请求
        #   - 如果引擎报告某请求完成，且 LMCache 也已完成 store 或正在处理，
        #     将其添加到 previously_finished
        #   - 否则添加到 ret_stores（返回给 Scheduler）
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)

        # Step 7: 计算最终完成的 store 请求
        #   - 取 finished_stores 和 previously_finished 的交集
        #   - 这是"安全"的完成请求（引擎和 LMCache 都知道的）
        ret_stores.update(self._update_and_get_finished_store())

        # Step 8: 返回结果
        #   - ret_stores: 已完成的 store 请求 ID
        #   - finished_retrieves: 已完成的 retrieve 请求 ID
        return ret_stores, finished_retrieves

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def shutdown(self):
        """
        Shutdown the LMCache MP worker adapter
        """
        logger.info("Unregistering kv caches")
        send_lmcache_request(
            self.mq_client, RequestType.UNREGISTER_KV_CACHE, [self.instance_id]
        ).result()

        self.mq_client.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str | None = None,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )

    def _create_hash_key(
        self, chunk_hash: bytes, request_id: str | None = None
    ) -> IPCCacheEngineKey:
        """Create a hash-mode IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=chunk_hash,
            request_id=request_id,
        )
