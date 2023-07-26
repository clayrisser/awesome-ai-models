pip install --quiet --upgrade pip

pip cache purge

python -m venv myenv

myenv\Scripts\activate # windows

source myenv/bin/activate # Linux

# Start with installing 'scipy'

pip install -vvv scipy

# Then, 'ftfy'

pip install -vvv ftfy

# Next, 'spacy' with the specified version 3.4.4

pip install -vvv spacy==3.4.4

# After that, 'diffusers'

pip install -vvv diffusers

# Next, 'transformers'

pip install -vvv transformers

# Then, 'accelerate'

pip install -vvv accelerate

# Finally, 'mediapy' and 'triton'

pip install -vvv mediapy triton
