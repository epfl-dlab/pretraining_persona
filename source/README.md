The python functions in this directory are mostly taken from the original repository. We applied the following changes to `eval_persona.py` to make it functional for base models:

- Added the revision argument to load specific checkpoints of the base models.
- Reduced default temperature to 0.5 and added a parameter.
- Added the repetition penalty parameter.
- Removed the chat template and concatenate instruction and prompt with a space.
- Do not address the model as an assistant.
- Added the few shot parameter to repeat examples in the prompt.
- Use a different coherence jugdement method that works for base models.