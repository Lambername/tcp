from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *
import sys

from torch import nn


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative

class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # src: the sequence to the encoder (required).
        # mask: the mask for the src sequence (optional).
        # src_key_padding_mask: the mask for the src keys per batch (optional).
        output = self.transformer_encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(CustomTransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: the sequence to the decoder (required).
        # memory: the sequence from the encoder, often called *memory* (required).
        # tgt_mask: the mask for the tgt sequence (optional).
        # memory_mask: the mask for the memory sequence (optional).
        # tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
        # memory_key_padding_mask: the mask for the memory keys per batch (optional).
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask, 
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output

class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		self.perception = resnet34(pretrained=True)

		self.measurements = nn.Sequential(
							nn.Linear(1+2+6, 128),
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)

		self.join_traj = nn.Sequential(
							nn.Linear(128+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.join_ctrl = nn.Sequential(
							nn.Linear(128+512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.speed_branch = nn.Sequential(
							nn.Linear(1000, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 1),
						)

		self.value_branch_traj = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		self.value_branch_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		# shared branches_neurons
		dim_out = 2

		self.policy_head = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)
		self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)
		self.output_ctrl = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
			)
		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())


		self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
		self.output_traj = nn.Linear(256, 2)

		self.init_att = nn.Sequential(
				nn.Linear(128, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)

		self.wp_att = nn.Sequential(
				nn.Linear(256+256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)

		self.merge = nn.Sequential(
				nn.Linear(512+256, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256),
			)
		
		# add
		self.trans_gru = nn.Linear(232*512, 256) # 要改
		self.trans_gru_ctr = nn.Linear(232*512, 256) # 要改
		self.fuse_up = nn.Linear(232, 256)
		self.fuse_down = nn.Linear(256, 232)
		# transformer组件
		self.query_embed = nn.Embedding(256, 512)
		self.trj_encoder = CustomTransformerEncoder(d_model=256, nhead=8, num_layers=6)   
		self.ctr_encoder = CustomTransformerEncoder(d_model=256, nhead=8, num_layers=6)
		self.ctr_decoder = CustomTransformerDecoder(d_model=256, nhead=8, num_layers=6)

		# GRU
		self.gru_trj = nn.GRU(input_size=256, hidden_size=640, num_layers=4, batch_first=True)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 , 1000)
	def forward(self, img, state, target_point):
		batch_size = img.shape[0]

		#image_shape(16,3,256,900)
		feature_emb, cnn_feature = self.perception(img)   # cnn_feature(batch_size,512,8,29)
		outputs = {}
		outputs['pred_speed'] = self.speed_branch(feature_emb)  # 从图像就发现速度？
		measurement_feature = self.measurements(state)

		

		# 调整 cnn_feature 的形状
		cnn_feature_trans = cnn_feature.reshape(batch_size, 512, -1)  # (batch_size, 512, 232)
		# cnn_feature_trans = cnn_feature.permute(0, 2, 3, 1)  # 现在形状是 (8, 8, 29, 512)

		# 扩展 vehicle_state 以匹配 cnn_feature 的空间维度
		# vehicle_state = measurement_feature.view(batch_size, 1, -1)  # 现在形状是 (8, 1, 1, 128)
		# vehicle_state = vehicle_state.expand(batch_size, 512, -1)  # 现在形状是 (8, 8, 29, 128)

		# 拼接特征
		# fused_features = torch.cat((cnn_feature_trans, vehicle_state), dim=-1)  # 现在形状是 (8, 512, 360)
		# fused_features = self.fuse_down(fused_features)   # (8, 8, 29, 512)

		# 调整 fused_features 以匹配 Transformer 的输入形状
		fused_features = cnn_feature_trans.permute(1, 0, 2)  # 现在形状是 (512, 8, 232)
		fused_features = self.fuse_up(fused_features)

		# 假设 self.transformer 是你的 Transformer 模型
		encoder_output = self.trj_encoder(fused_features)
		
		output = encoder_output.permute(1, 0, 2)
		output = self.fuse_down(output)
		# output = output.reshape(batch_size, 512, 8 -1)
		# output = self.avgpool(output)
		# output = torch.flatten(output, 1)
		print(output.shape)
        # x = self.fc(x)
		# output = output.reshape(batch_size, -1) #232*512
		# feature_extractor = self.trans_gru(output)

		# h0 = torch.zeros()
		# gru_waypoint, _ = self.gru_trj(feature_extractor.unsqueeze(1))
		# gru_waypoint = gru_waypoint.squeeze(1)
		# print("111111111111_gru_way")
		# print(gru_waypoint.shape)



		
		j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))   # j_traj(8, 256)

		# test = self.query_embed(test)
		

		outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
		outputs['pred_features_traj'] = j_traj


		z = j_traj
		output_wp = list()
		traj_hidden_state = list()

		# initial input variable to GRU
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

		# autoregressive generation of output waypoints
		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, target_point], dim=1)
			z = self.decoder_traj(x_in, z)
			traj_hidden_state.append(z)
			dx = self.output_traj(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)  # pred_wp（8，4，2）
		outputs['pred_wp'] = pred_wp

		decoder_output = self.ctr_encoder(fused_features)
		decoder_output = self.ctr_decoder(encoder_output, decoder_output)

		de_output = decoder_output.permute(1, 0, 2)
		de_output = de_output.reshape(-1, 232*640)
		j_ctrl = self.trans_gru_ctr(output)

		traj_hidden_state = torch.stack(traj_hidden_state, dim=1)

		# att = self.init_att[:-1](measurement_feature).view(-1, 1, 8, 29)
		init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)

		feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))

		#j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
		outputs['pred_features_ctrl'] = j_ctrl
		policy = self.policy_head(j_ctrl)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

		x = j_ctrl
		mu = outputs['mu_branches']
		sigma = outputs['sigma_branches']
		future_feature, future_mu, future_sigma = [], [], []
		wp_att_list = []

		# initial hidden variable to GRU
		h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, mu, sigma], dim=1)
			h = self.decoder_ctrl(x_in, h)
			wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
			new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
			merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
			dx = self.output_ctrl(merged_feature)
			x = dx + x

			policy = self.policy_head(x)
			mu = self.dist_mu(policy)
			sigma = self.dist_sigma(policy)
			future_feature.append(x)
			future_mu.append(mu)
			future_sigma.append(sigma)
			wp_att_list.append(wp_att)


		outputs['future_feature'] = future_feature
		outputs['future_mu'] = future_mu
		outputs['future_sigma'] = future_sigma

		outputs['wp_att'] = wp_att_list[2]

		return outputs

	def process_action(self, pred, command, speed, target_point):
		action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = action.cpu().numpy()[0].astype(np.float64)
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return steer, throttle, brake, metadata

	def _get_action_beta(self, alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

		x = x * 2 - 1

		return x

	def control_pid(self, waypoints, velocity, target):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata


	def get_action(self, mu, sigma):
		action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
		acc, steer = action[:, 0], action[:, 1]
		if acc >= 0.0:
			throttle = acc
			brake = torch.zeros_like(acc)
		else:
			throttle = torch.zeros_like(acc)
			brake = torch.abs(acc)

		throttle = torch.clamp(throttle, 0, 1)
		steer = torch.clamp(steer, -1, 1)
		brake = torch.clamp(brake, 0, 1)

		return throttle, steer, brake