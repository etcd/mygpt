import matplotlib.pyplot as plt


def plot_bigrams(bigrams, decode):
    plt.figure(figsize=(15, 15))
    plt.imshow(bigrams, cmap="Blues")
    for x in range(27):
        for y in range(27):
            plt.text(x, y, decode([y, x]),
                     ha='center', va='bottom', color="gray")
            plt.text(x, y, bigrams[y, x].item(),
                     ha='center', va='top', color="gray")
    plt.axis('off')
    plt.show()
