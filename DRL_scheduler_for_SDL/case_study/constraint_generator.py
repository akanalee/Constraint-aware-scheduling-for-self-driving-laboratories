"""
Constraint Generator - Tmax
SDL
"""

import numpy as np
from typing import List, Dict, Tuple


class ConstraintGenerator:
    """
    SDL Tmax
    /
    """

    # SDL
    CONSTRAINT_TEMPLATES = {
        'oxidation_sensitive': {
            # 
            'max_interval': 5.0,  # minutes
            'severity': 'critical',
            'penalty_factor': 100.0,
            'reason': 'Surface oxidation prevention'
        },
        'thermal_sensitive': {
            # 
            'max_interval': 10.0,
            'severity': 'high',
            'penalty_factor': 80.0,
            'reason': 'Thermal degradation prevention'
        },
        'moisture_sensitive': {
            # 
            'max_interval': 15.0,
            'severity': 'moderate',
            'penalty_factor': 50.0,
            'reason': 'Moisture absorption prevention'
        },
        'contamination_risk': {
            # 
            'max_interval': 30.0,
            'severity': 'low',
            'penalty_factor': 20.0,
            'reason': 'Cross-contamination prevention'
        }
    }

    @staticmethod
    def generate_sdl_constraints(process_steps: Dict) -> List[Dict]:
        """
        SDLTmax

        Args:
            process_steps: sdl_config.yamlprocess_steps

        Returns:
            constraints: List[{
                'from_step': str,
                'to_step': str,
                'max_interval': float,
                'reason': str,
                'severity': str,
                'penalty_factor': float
            }]
        """
        constraints = []

        # 1. S3 (Pre-reaction)  S4 (Reactor)
        # 
        constraints.append({
            'from_step': 'S3',
            'to_step': 'S4',
            'max_interval': 10.0,
            'reason': 'Prevent sample surface oxidation after pre-treatment',
            'severity': 'critical',
            'penalty_factor': 100.0
        })

        # 2. S4 (Reactor)  S6 (Spectroscopy)
        # /
        constraints.append({
            'from_step': 'S4',
            'to_step': 'S6',
            'max_interval': 15.0,
            'reason': 'Film quality degradation in air',
            'severity': 'critical',
            'penalty_factor': 100.0
        })

        # 3. S4 (Reactor)  S7 (Mass Spec)
        # 
        constraints.append({
            'from_step': 'S4',
            'to_step': 'S7',
            'max_interval': 15.0,
            'reason': 'Chemical composition stability',
            'severity': 'critical',
            'penalty_factor': 100.0
        })

        # 4. S2 (Transfer)  S3 (Pre-reaction)
        # 
        constraints.append({
            'from_step': 'S2',
            'to_step': 'S3',
            'max_interval': 15.0,
            'reason': 'Solution evaporation on substrate',
            'severity': 'moderate',
            'penalty_factor': 50.0
        })

        # 5. S6 (Spectroscopy)  S7 (Mass Spec)
        # 
        constraints.append({
            'from_step': 'S6',
            'to_step': 'S7',
            'max_interval': 30.0,
            'reason': 'Cross-contamination prevention',
            'severity': 'low',
            'penalty_factor': 20.0
        })

        return constraints

    @staticmethod
    def generate_random_constraints(
            n_steps: int,
            n_constraints: int,
            min_interval: float = 5.0,
            max_interval: float = 30.0,
            seed: int = None
    ) -> List[Dict]:
        """
        Tmax

        Args:
            n_steps: 
            n_constraints: 
            min_interval: 
            max_interval: 
            seed: 

        Returns:
            constraints: 
        """
        if seed is not None:
            np.random.seed(seed)

        constraints = []
        used_pairs = set()

        for _ in range(n_constraints):
            # from < to
            while True:
                from_idx = np.random.randint(0, n_steps - 1)
                to_idx = np.random.randint(from_idx + 1, n_steps)
                pair = (from_idx, to_idx)

                if pair not in used_pairs:
                    used_pairs.add(pair)
                    break

            # 
            interval = np.random.uniform(min_interval, max_interval)

            # 
            severity = np.random.choice(['critical', 'high', 'moderate', 'low'])
            penalty_factors = {'critical': 100.0, 'high': 80.0, 'moderate': 50.0, 'low': 20.0}

            constraints.append({
                'from_step': f'S{from_idx + 1}',
                'to_step': f'S{to_idx + 1}',
                'max_interval': round(interval, 1),
                'reason': f'Random constraint {len(constraints) + 1}',
                'severity': severity,
                'penalty_factor': penalty_factors[severity]
            })

        return constraints

    @staticmethod
    def validate_constraints(
            constraints: List[Dict],
            process_steps: Dict
    ) -> Tuple[bool, List[str]]:
        """
        

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        step_names = set(process_steps.keys())

        for i, constraint in enumerate(constraints):
            from_step = constraint['from_step']
            to_step = constraint['to_step']

            # 
            if from_step not in step_names:
                errors.append(f"Constraint {i}: from_step '{from_step}' not found")
            if to_step not in step_names:
                errors.append(f"Constraint {i}: to_step '{to_step}' not found")

            # 
            if constraint['max_interval'] <= 0:
                errors.append(f"Constraint {i}: max_interval must be positive")

            # fromto
            step_order = list(process_steps.keys())
            if from_step in step_order and to_step in step_order:
                if step_order.index(from_step) >= step_order.index(to_step):
                    errors.append(f"Constraint {i}: from_step must precede to_step in process")

        return len(errors) == 0, errors