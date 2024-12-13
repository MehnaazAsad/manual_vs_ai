# Manual vs AI
This repo contains all the code for both manual and AI-assisted approaches in this comparison study of detecting cheaters in CS:GO. Detailed performance metrics along with the guidance document and prompts can be found here too.

### Contents in `/codes` :computer: 

| Filename                           | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| `manual_baseline.ipynb`            | Baseline model in manual approach Phase I                    |
| `manual_feature_engineering.ipynb` | Slightly more advanced feature engineering in manual approach Phase II |
| `manual_lstm_idea.ipynb`           | LSTM idea in manual approach (which wasn't implemented due to computational constraints) |
| `AI_phase_I_randomforest.ipynb`    | Random forest implementation from unstructured AI-assisted approach Phase I |
| `AI_phase_II_lstm.ipynb`           | LSTM implementation from structured AI-assisted approach Phase II |
| `AI_phase_II.ipynb`                | Main AI-assisted code from Phase II with logistic regression and random forest implementations |
| `firing_pattern_viz.ipynb`         | Visualizing firing patterns of guns used in CS:GO            |

### Contents in `/prompts` :abc:

| Filename                       | Description                                               |
| ------------------------------ | --------------------------------------------------------- |
| `guidance_creation_prompt.png` | Initial prompt used to create guidance document           |
| `implementation_prompt.md`     | Final prompt used when implementing the guidance document |

### Additional contents :file_cabinet:

| Filename               | Description                            |
| ---------------------- | -------------------------------------- |
| `metrics.md`           | All performance metrics                |
| `ml_beginner_guide.md` | Final version of the guidance document |

