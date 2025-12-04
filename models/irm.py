import torch
import torch.nn as nn
import torch.nn.functional as F

class IterativeRefinementModel(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim=256, chunk_size=16, train_noise_std=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.train_noise_std = train_noise_std

        # 観測エンコーダ
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Refinement Network
        self.refine_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim * chunk_size + 32, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim * chunk_size)
        )
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

    def forward_refine(self, obs_emb, current_action_guess, k):
        B = obs_emb.shape[0]
        k_tensor = torch.full((B, 1), float(k), device=obs_emb.device)
        t_emb = self.time_mlp(k_tensor)
        act_flat = current_action_guess.view(B, -1)
        
        x = torch.cat([obs_emb, act_flat, t_emb], dim=-1)
        
        residual = self.refine_net(x)
        refined_action = act_flat + residual
        
        return refined_action.view(B, self.chunk_size, -1)

    def compute_loss(self, batch):
        obs = batch['obs']
        target_action = batch['action']
        B = obs.shape[0]
        obs_emb = self.obs_encoder(obs)
        total_loss = 0
        
        # --- 修正点1: Cold/Warm の混合学習 (Distribution Matching) ---
        # 推論時の初回(Cold)と2回目以降(Warm)の両方に対応させるため、確率的に切り替える
        # アクションは [-1, 1] に正規化されていると仮定し、ゼロ初期化を使う
        
        if torch.rand(1).item() < 0.5:
            # Cold Start: ゼロから正解を当てるタスク
            current_guess = torch.zeros_like(target_action)
        else:
            # Warm Start Simulation: 正解をずらした位置から修正するタスク
            shifted_action = target_action.clone()
            shifted_action[:, :-1, :] = target_action[:, 1:, :]
            shifted_action[:, -1, :] = target_action[:, -1, :]
            
            # ここでのノイズは「前回の予測誤差」を模倣するもの
            warm_noise = torch.randn_like(shifted_action) * self.train_noise_std
            current_guess = shifted_action + warm_noise
        
        num_refine_steps = 2 
        
        for k in range(num_refine_steps):
            # C2: Stochasticity Injection
            # 入力にノイズを乗せることで、多様体への吸着(Manifold Adherence)を学習させる
            noise = torch.randn_like(current_guess) * self.train_noise_std
            noisy_input = current_guess + noise
            
            # Refine
            refined_pred = self.forward_refine(obs_emb, noisy_input, k)
            
            # Loss計算 (C3: Supervised Iterative Compute)
            # 各ステップで正解に近づくよう監督する
            loss_k = F.mse_loss(refined_pred, target_action)
            total_loss += loss_k
            
            # --- 修正点2: ループ更新ロジックの修正 ---
            # 「予測された結果」を次のステップの初期値にする。
            # detach()することで、各ステップを独立した回帰問題として解かせる(MIP推奨)
            current_guess = refined_pred.detach()
            
        return total_loss

    @torch.no_grad()
    def predict_with_warm_start(self, obs, prev_action_chunk=None):
        B = obs.shape[0]
        obs_emb = self.obs_encoder(obs)
        
        # --- Warm Start Strategy ---
        if prev_action_chunk is None:
            # 学習時のCold Startと合わせるため、0.5ではなく0.0 (正規化前提) を推奨
            current_guess = torch.zeros((B, self.chunk_size, self.action_dim)).to(obs.device)
        else:
            current_guess = prev_action_chunk.clone()
            current_guess[:, :-1, :] = prev_action_chunk[:, 1:, :]
            current_guess[:, -1, :] = prev_action_chunk[:, -1, :]
            
        # Refinement Loop
        num_refine_steps = 2
        for k in range(num_refine_steps):
            # 推論時はノイズなしでRefine
            current_guess = self.forward_refine(obs_emb, current_guess, k)
            
        return current_guess