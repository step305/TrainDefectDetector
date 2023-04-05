from backend import rcnn_train_model
import tkinter
from tkinter import filedialog as file_dlg


if __name__ == '__main__':
    root = tkinter.Tk()
    dir_path = file_dlg.askdirectory(mustexist=True)
    root.update()
    root.destroy()
    train_maskrcnn = rcnn_train_model.RCNNTraining()
    train_maskrcnn.load_model("mask_rcnn_coco.h5")
    train_maskrcnn.load_dataset(dir_path)
    train_maskrcnn.train_model(num_epochs=300, path_trained_models="mask_rcnn_models")
