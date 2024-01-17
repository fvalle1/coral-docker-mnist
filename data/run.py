
# ## Coral
# %%
#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#import keras
#import keras.backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import os
os.chdir("/home/data/")


X_test = np.loadtxt("X_test.txt")
Y_test = np.loadtxt("Y_test.txt")
  
# ! curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# ! echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
# ! sudo apt-get update
# ! sudo apt-get install edgetpu-compiler	
#! edgetpu_compiler $TFLITE_FILE -s

# %%
import os
os.environ["DYLD_LIBRARY_PATH"]="/usr/local/lib"

# %%
interpreter = make_interpreter('model_edgetpu.tflite', device=":0")
interpreter.allocate_tensors()
width, height = common.input_size(interpreter)


# %%
classify.get_classes(interpreter, top_k=1)

# %%
def pred(X_data):
    common.set_input(interpreter, X_data.reshape((width, height, 1))/255.)
    interpreter.invoke()
    return classify.get_classes(interpreter, top_k=1)[0].id

y_pred = [pred(x_test) for x_test in X_test.reshape(-1,28,28,1)]
y_real = Y_test

# %%
cm = sns.heatmap(
    confusion_matrix(y_real, y_pred, normalize="true"),
    )

# %%



