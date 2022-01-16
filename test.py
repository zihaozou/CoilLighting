import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

# plot sin wave
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(-np.pi, np.pi)

ax.set_xlim(-np.pi, np.pi)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.plot(x, np.sin(x), label="sin")

ax.legend()
ax.set_title("sin(x)")


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# you can get a high-resolution image as numpy array!!
plot_img_np = get_img_from_fig(fig)
print(plot_img_np.shape)
