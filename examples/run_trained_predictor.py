"""Example showing how to use predictor_call.

This hard-codes paths to the latest trained models.
"""

# Hacky way to import parent directory modules
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictor_call


def main():
    # Load model
    predictor = predictor_call.Predictor(
        'models/cas13/classify/model-51373185/serialized',
        'models/cas13/regress/model-f8b6fd5d/serialized')

    # As input, use list of tuples giving (target with 10 nt context, guide)
    pairs = [
            ('A'*10 + 'A'*28 + 'A'*10, 'A'*28),
            ('A'*10 + 'A'*28 + 'A'*10, 'A'*12 + 'GCTG' + 'A'*12)
            ]

    # Start position is irrelevant; make it -1
    predictions = predictor.compute_activity(-1, pairs)

    # Print predictions
    for pair, o in zip(pairs, predictions):
        print(pair, o)


if __name__ == "__main__":
    main()
