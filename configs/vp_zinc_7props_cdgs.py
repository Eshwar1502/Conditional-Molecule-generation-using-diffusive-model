"""Training GNN on ZINC250k with continuous VPSDE."""

import ml_collections
import torch



def get_config():
    config = ml_collections.ConfigDict()

    config.model_type = 'mol_sde'

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vpsde'
    training.continuous = True
    training.reduce_mean = False

    training.batch_size = 16
    training.eval_batch_size = 8
    training.n_iters = 2000000
    training.snapshot_freq = 5000  # SET Larger values to save less checkpoints
    training.log_freq = 200
    training.eval_freq = 5000
    ## store additional checkpoints for preemption
    training.snapshot_freq_for_preemption = 2000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    # sampling.method = 'pc'
    sampling.method = 'dpm3'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'
    sampling.rtol = 1e-5
    sampling.atol = 1e-5
    sampling.ode_method = 'rk4'
    # sampling.ode_step = 0.01
    sampling.ode_step = 100

    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.atom_snr = 0.16
    sampling.bond_snr = 0.16
    sampling.vis_row = 4
    sampling.vis_col = 4
    
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             

    sampling.property_scale = 1.0  # Strength of property conditioning
    # sampling.property_noise = 0.0 

# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

    
    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 15
    evaluate.end_ckpt = 40
    evaluate.batch_size = 8  # 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 5000
    evaluate.mmd_distance = 'RBF'
    evaluate.max_subgraph = False
    evaluate.save_graph = True
    evaluate.nn_eval = False
    evaluate.nspdk = False
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             

    # evaluate.property_targets = []  # Will be set via command line
    # evaluate.property_targets = dict(property_targets=[3.5, 2.1, 0.8, 6.2, 85.0, 0.6, 4.3])
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

    # data
    config.debug_mode = False
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = 'data'
    data.name = 'ZINC250k_7props'

    # data.property = ['molecular-weight', 'volume', 'density', 'nha', 'nhd', 'tpsa', 'nrot', 'nring', 'maxring', 'nhet', 'fchar', 'nrig', 'flexibility', 'stereo-centers', 'gasa', 'qed', 'sascore', 'fsp3', 'mce-18', 'npscore', 'alarm_nmr-rule', 'bms-rule', 'chelating-rule', 'pains', 'lipinski-rule', 'pfizer-rule', 'gsk-rule', 'goldentriangle', 'logs', 'logd_7.4', 'logp', 'melting-point', 'boiling-point', 'pka', 'pkb', 'caco-2-permeability', 'mdck-Permeability', 'pampa', 'pgp-inhibitor', 'pgp-substrate', 'hia', 'f20', 'f30', 'f50', 'oatp1b1-inhibitor', 'oatp1b3-inhibitor', 'bcrp-inhibitor', 'bsep-inhibitor', 'bbb', 'mrp1-inhibitor', 'ppb', 'vdss', 'fu', 'cyp1a2-inhibitor', 'cyp1a2-substrate', 'cyp2c19-inhibitor', 'cyp2c19-substrate', 'cyp2c9-inhibitor', 'cyp2c9-substrate', 'cyp2d6-inhibitor', 'cyp2d6-substrate', 'cyp3a4-inhibitor', 'cyp3a4-substrate', 'cyp2b6-inhibitor', 'cyp2b6-substrate', 'cyp2c8-inhibitor', 'hlm-stability', 'clplasma', 't1/2', 'bcf', 'igc50', 'lc50dm', 'lc50fm', 'herg-blockers', 'herg-blockers-10um', 'dili', 'ames-toxicity', 'rat-oral-acute-toxicity', 'fdamdd', 'skin-sensitization', 'carcinogenicity', 'eye-corrosion', 'eye-irritation', 'respiratory', 'human-hepatotoxicity', 'drug-induced-neurotoxicity', 'ototoxicity', 'hematotoxicity', 'drug-induced-nephrotoxicity', 'genotoxicity', 'rpmi-8226-immunitoxicity', 'a549-cytotoxicity', 'hek293-cytotoxicity', 'nr-ahr', 'nr-ar', 'nr-ar-lbd', 'nr-aromatase', 'nr-er', 'nr-er-lbd', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53', 'nonbiodegradable', 'nongenotoxic_carcinogenicity', 'surechembl', 'ld50_oral', 'skin_sensitization', 'acute_aquatic_toxicity', 'faf-drugs4-rule', 'genotoxic_carcinogenicity_mutagenicity', 'colloidal-aggregators', 'fluc-inhibitors', 'blue-fluorescence', 'green_fluorescence', 'reactive-compounds', 'Other_assay_interference']
    
    data.property = [ 'gasa', 'qed', 'sascore', 'fsp3', 'mce-18', 'npscore', 'alarm_nmr-rule',    'bms-rule', 'chelating-rule', 'pains', 'lipinski-rule', 'pfizer-rule', 
    'gsk-rule', 'goldentriangle', 'colloidal-aggregators', 'fluc-inhibitors', 
    'blue-fluorescence', 'green_fluorescence', 'reactive-compounds',
    'caco-2-permeability', 'mdck-Permeability', 'pampa', 'pgp-inhibitor', 
    'pgp-substrate', 'hia', 'f20', 'f30', 'f50',
    'ppb', 'vdss', 'fu', 'bbb', 'oatp1b1-inhibitor', 'oatp1b3-inhibitor', 
    'bcrp-inhibitor', 'mrp1-inhibitor'
]
    data.num_properties = len(data.property)   # Number of properties
    data.norm_properties = True


    
    data.split_ratio = 0.8
    data.max_node = 38
    data.atom_channels = 9
    data.bond_channels = 2
    data.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    data.norm = (0.5, 1.0)



    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'CDGS'
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    # model.nonlinearity = 'elu'
    model.nf = 256
    # model.nf = 384
    model.num_gnn_layers = 10
    # model.num_gnn_layers = 8
    model.conditional = True
    model.embedding_type = 'positional'
    model.rw_depth = 20
    model.graph_layer = 'GINE'
    model.edge_th = -1.
    model.heads = 8
    model.dropout = 0.1

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             

    model.property_guidance = 'cross-attention' 
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

    model.num_properties = data.num_properties
    model.property_dim = 128 # New: dedicated embedding dimension
    model.property_attention = True  # Enable cross-attention

    model.num_scales = 1000  # SDE total steps (N)
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.node_beta_min = 0.1
    model.node_beta_max = 20.
    model.edge_beta_min = 0.1
    model.edge_beta_max = 20.

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.  # SET Larger values to converge faster, e.g., 10.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
