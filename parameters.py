r_UPD = [1] #[1, 5, 10]
r_BUFFER_SIZE = [1e5] #[1e5, 5e5, 1e6]
r_BATCH_SIZE = [64] #[64, 128, 256, 512]


r_fc1_units = [300] #[200, 300, 400]
r_fc2_units = [400] #[200, 300, 400]

#Actor parameters
r_LR_ACTOR = [1e-3] #[1e-4, 5e-4, 1e-3]
r_a_gradient_clipping = [False] #[True, False]
r_a_leaky = [True] #[True, False]
r_a_dropout = [False] #[True, False]

#Critic parameters
r_LR_CRITIC = [1e-4] #[1e-4, 5e-4, 1e-3]
r_c_gradient_clipping = [False] #[True, False]
r_c_batch_norm = [True] #[True, False]
r_c_leaky = [True] #[True, False]
r_c_dropout = [False] #[True, False]
