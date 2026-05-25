tune_params = {
    'epochs': {
        'value': 100
    },
    'batch_size': {
        'values': [32, 64]
    },
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0.0001,
        'max': 0.01
    }
}

objective = {
    'name': 'val_acc',
    'goal': 'maximize'
}

training = {
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'metrics': ['accuracy']
}

mqtt_callback_api = 'mqtt.CallbackAPIVersion.VERSION2'

# Default data config
window_size     = 3
step_size       = {'mhealth': 1.6, 'pamap2':1.96, 'dsads':1.6, 'our_cont':0.5}
sensor_types    = ['accel']
positions       = ['hand', 'ankle']
b = 50  # the number of evaluation points for KDE
h = 0.1 # Bandwidth for KDE
sampling    = 50
is_rotate   = True
wo_fe       = False

activities = [
    'standing', 
    'sitting',
    'ascending_stairs',
    'walking',
    'descending_stairs',
]

color_x = '#EF553B'
color_y = '#00CC96'
color_z = '#636EFA'

method  = 'supervised'
seed    = 42

hyperparams_config = {
    'baseline':
        {
            'weight_decay' : 0.001,
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 60
        },
    'bams':
        {
            'weight_decay' : 0.1,
            'batch_size': 128,
            'lr': 3e-4,
            'epochs': 80
        },
    'cosar':
        {
            'weight_decay' : 0.05,
            'batch_size': 256,
            'lr': 1e-4,
            'epochs': 100
        },
    'llm4har': {
        'weight_decay' : 0.01,
        'batch_size'   : 32,    # smaller batch
        'lr'           : 1e-2,  # reduce from 1e-4 → 1e-5
        'epochs'       : 200
    }
}

