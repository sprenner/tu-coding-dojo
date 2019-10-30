import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, hamming_loss
from sklearn.utils.multiclass import unique_labels



def pplot_cm(y_true, y_pred, normalize=False, large=False, font_scale=1):
    """Function is similar to sklearn's confusion_matrix, only prettier. Set normalize=True for percentages. Set large=True for larger figure. Set font_size to float > 0 to scale font size.
        """
        
    labels = sorted(unique_labels(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm/cm.sum(axis=1)

    df = pd.DataFrame(cm, index=labels, columns=labels)

    sns.set(font_scale=font_scale,
            style="whitegrid",
            context="poster" if large else "notebook",)
    chart = sns.heatmap(df, annot=True)
    chart.set(xlabel='Predicted Labels', ylabel='True Labels', title="Confusion Matrix")
    chart.set_xticklabels(labels, rotation=60)
    plt.show()

    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1:', f1_score(y_true, y_pred, average='weighted'))
    print('Precision:', precision_score(y_true, y_pred, average='weighted'))
    print('Hamming loss', hamming_loss(y_true, y_pred))

    return
