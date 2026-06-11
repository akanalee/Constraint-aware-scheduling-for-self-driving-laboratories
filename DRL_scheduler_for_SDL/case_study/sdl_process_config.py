"""

CVD
"""

PROCESS_DEFINITION = {
    'S1_substrate_prep': {
        'name': 'Substrate Preparation',
        'operations': ['wash', 'dry', 'clean'],
        'machines': {
            'washer_A': {'capacity': 1, 'type': 'ultrasonic'},
            'washer_B': {'capacity': 1, 'type': 'ultrasonic'},
            'dryer': {'capacity': 2, 'type': 'N2_blow'}
        },
        'processing_time': {
            'base': 2.5,  # minutes
            'std': 0.5,  # ±20%
            'distribution': 'normal'
        }
    },

    'S2_robotic_transfer': {
        'name': 'Robotic Transfer',
        'operations': ['pickup', 'transfer', 'place'],
        'machines': {
            'robot_arm_A': {'capacity': 1, 'dof': 6},
            'robot_arm_B': {'capacity': 1, 'dof': 6}
        },
        'processing_time': {
            'base': 1.0,
            'std': 0.1,
            'distribution': 'normal'
        },
        'transfer_matrix': {  # -
            ('S1', 'S3'): 1.0,
            ('S3', 'S4'): 1.5,
            ('S4', 'S5'): 0.8,
            ('S5', 'S6'): 1.2,
            ('S6', 'S7'): 1.0
        }
    },

    'S3_pre_reaction': {
        'name': 'Pre-reaction Treatment',
        'operations': ['solution_immersion', 'thermal_treatment', 'plasma_clean', 'anneal'],
        'machines': {
            'solution_bath_A': {'capacity': 3, 'type': 'chemical'},
            'solution_bath_B': {'capacity': 3, 'type': 'chemical'},
            'heater': {'capacity': 1, 'max_temp': 500},
            'plasma_cleaner': {'capacity': 1, 'power': '100W'},
            'annealer': {'capacity': 2, 'atmosphere': 'N2'}
        },
        'processing_time': {
            'base': 5.0,
            'std': 1.5,
            'distribution': 'normal'
        }
    },

    'S4_reactor': {
        'name': 'CVD/ALD Reactor',
        'operations': ['load', 'evacuate', 'deposit', 'cool_in_chamber'],
        'machines': {
            'CVD_reactor': {
                'capacity': 1,
                'type': 'thermal_CVD',
                'precursors': ['TMGa', 'AsH3'],
                'atlas_controlled': True  # 
            },
            'ALD_reactor': {
                'capacity': 1,
                'type': 'plasma_ALD',
                'precursors': ['TMA', 'H2O'],
                'atlas_controlled': True
            }
        },
        'processing_time': {
            'base': 11.0,
            'std': 2.75,  # ±25%
            'distribution': 'atlas_modulated',  # 
            'atlas_params': {
                'target_thickness': 100,  # nm
                'growth_rate_range': (8, 12),  # nm/min
                'optimization_cycles': 50
            }
        }
    },

    'S5_cooling': {
        'name': 'Post-reaction Cooling',
        'operations': ['controlled_cooldown', 'stabilization'],
        'machines': {
            'cooling_chamber_A': {'capacity': 2, 'cooling_rate': 5},  # °C/min
            'cooling_chamber_B': {'capacity': 2, 'cooling_rate': 5}
        },
        'processing_time': {
            'base': 3.5,
            'std': 0.7,
            'distribution': 'normal'
        }
    },

    'S6_spectroscopy': {
        'name': 'Spectroscopy Characterization',
        'operations': ['mount_sample', 'calibrate', 'measure', 'data_export'],
        'machines': {
            'UV_Vis': {'capacity': 1, 'wavelength_range': (200, 800)},
            'FTIR': {'capacity': 1, 'resolution': 4},
            'Raman': {'capacity': 1, 'laser': '532nm'}
        },
        'processing_time': {
            'base': 4.0,
            'std': 0.8,
            'distribution': 'normal'
        }
    },

    'S7_mass_spec': {
        'name': 'Mass Spectrometry',
        'operations': ['sample_prep', 'ionization', 'analysis', 'cleanup'],
        'machines': {
            'ICP_MS': {
                'capacity': 1,
                'type': 'quadrupole',
                'is_bottleneck': True  # 
            }
        },
        'processing_time': {
            'base': 6.0,
            'std': 1.8,
            'distribution': 'normal'
        }
    }
}

# Tmax
TMAX_CONSTRAINTS = [
    {
        'from_step': 'S3',
        'to_step': 'S4',
        'max_interval': 10.0,  # minutes
        'reason': 'Prevent sample surface oxidation',
        'severity': 'critical',
        'penalty_factor': 100.0
    },
    {
        'from_step': 'S4',
        'to_step': 'S6',
        'max_interval': 15.0,
        'reason': 'Film quality degradation in air',
        'severity': 'critical',
        'penalty_factor': 100.0
    },
    {
        'from_step': 'S4',
        'to_step': 'S7',
        'max_interval': 15.0,
        'reason': 'Chemical composition stability',
        'severity': 'critical',
        'penalty_factor': 100.0
    },
    {
        'from_step': 'S2',
        'to_step': 'S3',
        'max_interval': 15.0,
        'reason': 'Solution evaporation on substrate',
        'severity': 'moderate',
        'penalty_factor': 50.0
    },
    {
        'from_step': 'S6',
        'to_step': 'S7',
        'max_interval': 30.0,
        'reason': 'Cross-contamination prevention',
        'severity': 'low',
        'penalty_factor': 20.0
    }
]

# Setup Time/
MACHINE_LAYOUT = {
    'S1': {'x': 0, 'y': 0},
    'S2_robots': {'x': 5, 'y': 0},
    'S3': {'x': 10, 'y': 0},
    'S4_reactors': {'x': 15, 'y': 0},
    'S5': {'x': 15, 'y': 5},
    'S6': {'x': 10, 'y': 5},
    'S7': {'x': 5, 'y': 5}
}


def compute_transfer_time(from_step, to_step):
    """
    
     = 2 m/min
    """
    pos1 = MACHINE_LAYOUT[from_step]
    pos2 = MACHINE_LAYOUT[to_step]
    distance = ((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2) ** 0.5
    return distance / 2.0  # 


# Atlas
ATLAS_CONFIG = {
    'optimization_objective': 'target_thickness',
    'parameter_space': {
        'temperature': (400, 600),  # °C
        'pressure': (1, 10),  # Torr
        'flow_rate': (10, 100),  # sccm
        'time': (5, 20)  # minutes
    },
    'exploration_schedule': {
        'phase_1': {  # 10jobs
            'jobs_range': (0, 10),
            'param_variation': 0.25  # ±25%
        },
        'phase_2': {  # 10-50jobs
            'jobs_range': (10, 50),
            'param_variation': 0.10
        },
        'phase_3': {  # 50+jobs
            'jobs_range': (50, 200),
            'param_variation': 0.05
        }
    },
    'learning_model': 'gaussian_process',  # GP
    'seed': 42
}

# Job
JOB_TYPES = {
    'GaAs_deposition': {
        'sequence': ['S1', 'S2', 'S3', 'S4_CVD', 'S5', 'S2', 'S6_UV', 'S2', 'S7'],
        'material': 'GaAs',
        'substrate': 'Si',
        'priority': 'normal'
    },
    'Al2O3_ALD': {
        'sequence': ['S1', 'S2', 'S3', 'S4_ALD', 'S5', 'S2', 'S6_FTIR', 'S2', 'S7'],
        'material': 'Al2O3',
        'substrate': 'glass',
        'priority': 'normal'
    },
    'InP_urgent': {
        'sequence': ['S1', 'S2', 'S3', 'S4_CVD', 'S5', 'S2', 'S6_Raman', 'S2', 'S7'],
        'material': 'InP',
        'substrate': 'Si',
        'priority': 'high'  # 
    }
}

# 
EXPERIMENT_SCALES = {
    'debug': {
        'n_jobs': 5,
        'load_factor': 0.5,
        'job_types': ['GaAs_deposition'],
        'purpose': 'maskreward'
    },
    'small': {
        'n_jobs': 20,
        'load_factor': 0.6,
        'job_types': ['GaAs_deposition', 'Al2O3_ALD'],
        'purpose': 'MILP + BC'
    },
    'medium': {
        'n_jobs': 50,
        'load_factor': 0.8,
        'job_types': ['GaAs_deposition', 'Al2O3_ALD', 'InP_urgent'],
        'purpose': ''
    },
    'large': {
        'n_jobs': 200,
        'load_factor': 0.9,
        'job_types': ['GaAs_deposition', 'Al2O3_ALD', 'InP_urgent'],
        'purpose': ' + '
    },
    'stress_test': {
        'n_jobs': 500,
        'load_factor': 0.95,
        'job_types': ['GaAs_deposition', 'Al2O3_ALD', 'InP_urgent'],
        'purpose': ''
    }
}