import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler, SliceSampler


class Buffer:
    def __init__(self, config, curious_replay=None):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        max_size = int(config.max_size)

        self.use_curious_replay = bool(curious_replay is not None and curious_replay.enabled)
        if self.use_curious_replay:
            self.cr_c = float(curious_replay.c)
            self.cr_beta = float(curious_replay.beta)
            self.cr_alpha = float(curious_replay.alpha)
            self.cr_eps = float(curious_replay.eps)
            self.cr_p_max = float(curious_replay.p_max)
            sampler = PrioritizedSliceSampler(
                max_capacity=max_size,
                alpha=1.0,  # the CR formula already supplies the sharpness via cr_alpha
                beta=0.0,  # disable importance-sampling weights (CR doesn't use them)
                eps=0.0,
                num_slices=self.batch_size,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
                strict_length=True,
            )
            # Visit-count tensor parallel to the SumTree (1D, capacity = max_size * env_num,
            # but env_num=1 in current configs so this matches max_size).
            self._visit_counts = torch.zeros(max_size, dtype=torch.float32, device="cpu")
        else:
            sampler = SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            )
            self._visit_counts = None

        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=max_size, device=self.storage_device, ndim=2),
            sampler=sampler,
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        indices = self._buffer.extend(data.unsqueeze(1))
        if self.use_curious_replay and indices is not None:
            # Reset visit counts at any positions just (re)written by extend so that
            # a freshly overwritten transition starts from v=0.
            flat = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
            self._visit_counts.index_fill_(0, flat.to(self._visit_counts.device), 0.0)

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def update_priority(self, index, transition_loss):
        """Update Curious Replay priorities for the just-sampled transitions.

        Priority follows Eq. (1) of Kauvar & Doyle et al. (2023):
            p_i = c * beta**v_i + (|L_i| + eps)**alpha
        clamped to ``p_max``. ``v_i`` is the per-transition visit count.
        """
        if not self.use_curious_replay:
            return
        storage_shape = self._buffer.storage.shape
        if storage_shape is None or storage_shape.numel() == 0:
            return
        # index is [time_idx (B, T), env_idx (B, T)] from `sample()`.
        time_idx = index[0].reshape(-1).to("cpu", dtype=torch.long)
        env_idx = index[1].reshape(-1).to("cpu", dtype=torch.long)
        env_num = int(storage_shape[1])
        # Flatten (time, env) -> 1D index matching the SumTree's capacity.
        flat = (time_idx * env_num + env_idx).contiguous()

        ones = torch.ones_like(flat, dtype=torch.float32)
        # scatter_add_ accumulates duplicates (a slice may revisit the same transition once).
        self._visit_counts.scatter_add_(0, flat, ones)
        v = self._visit_counts.gather(0, flat)

        loss_flat = transition_loss.detach().reshape(-1).abs().to("cpu", dtype=torch.float32).contiguous()
        priority = self.cr_c * (self.cr_beta**v) + (loss_flat + self.cr_eps).pow(self.cr_alpha)
        priority.clamp_(max=self.cr_p_max)

        self._buffer._sampler.update_priority(flat, priority)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
