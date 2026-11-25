"""Configuration for the simplified public training example."""


class Config:
    profiles = {
        'road8': ('road', 'arrow', 'lane', 'dashed', 'zebra', 'text', 'diamond', 'other'),
        'toronto6': ('road', 'arrow', 'lane', 'dashed', 'zebra', 'other'),
    }
    features = ('X', 'Y', 'Z', 'R', 'G', 'B', 'Intensity')
    in_features_dim = len(features)
    num_points = 8192
    num_neighbors = 16
    levels = (8192, 2048, 512, 128)
    num_layers = len(levels)
    d_out = (32, 64, 128, 256)
    dropout_rate = 0.5
    batch_size = 4
    max_epoch = 100
    learning_rate = 1e-3
    min_learning_rate = 1e-5
    lr_schedule = 'cosine'
    beta_1 = 0.9
    beta_2 = 0.999
    focal_gamma = 2.0
    weight_clip = (0.5, 5.0)
    ignore_label = -1

    def __init__(self, profile='road8'):
        if profile not in self.profiles:
            raise ValueError(f'Unknown profile: {profile}')
        self.profile = profile
        self.label_names = self.profiles[profile]
        self.num_classes = len(self.label_names)
        self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
        self.idx_to_label = dict(enumerate(self.label_names))
