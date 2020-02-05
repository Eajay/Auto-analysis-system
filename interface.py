from tkinter import *
from tkinter import filedialog
import readData
import ml
import ctypes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class Interface:
    def __init__(self, master):
        self.master = master
        self.master.title("Auto Analysis")
        self.all_data = readData.DataPreview()
        self.ml = ml.MachineLearning()

        self.filename = ""
        self.submit = False

        file_frame = Frame(self.master)
        file_frame.grid(row=0, column=0)
        self._btn_open_file = Button(file_frame, width=16, text="Choose local file",
                                     command=lambda: self._open_file())
        self._btn_open_file.pack(side=LEFT, padx=10, pady=10)

        self._text_filename = Text(file_frame, height=1, state=DISABLED)
        self._text_filename.pack(side=LEFT, padx=10, pady=10)

        self._btn_submit_file = Button(file_frame, width=16, text="Submit",
                                       command=lambda: self._submit_data())
        self._btn_submit_file.pack(side=LEFT, padx=10, pady=10)

        mode_frame = Frame(self.master)
        mode_frame.grid(row=1, column=0)
        mode = {"Web: plotly": "plotly", "Gui: matplotly": "matplotly"}
        self.v = StringVar(value="plotly")

        self._rbtn_web = Radiobutton(mode_frame, text="Web: plotly", variable=self.v, value=mode["Web: plotly"], command=lambda: self._send_text("Choosing plotly"))
        self._rbtn_web.pack(side=LEFT, padx=10)

        self._rbtn_gui = Radiobutton(mode_frame, text="Gui: matplotly", variable=self.v, value=mode["Gui: matplotly"], command=lambda: self._send_text("Choosing matplotly"))
        self._rbtn_gui.pack(side=LEFT, padx=10)


        preview_frame = Frame(self.master)
        preview_frame.grid(row=2, column=0, sticky=N, pady=50)

        self._label_preview = Label(preview_frame, text="Preview Chart", fg='blue', font=("Helvetica", 16))
        self._label_preview.grid(row=0, column=0, columnspan=3)

        self._btn_basic_table = Button(preview_frame, width=26, text="Basic Feature Table", command=lambda: self._basic_feature(showing=1))
        self._btn_basic_table.grid(row=1, column=0, padx=100, pady=50)

        self._btn_label_pie = Button(preview_frame, width=26, text="Label Pie", command=lambda: self._label_pie(showing=1))
        self._btn_label_pie.grid(row=1, column=1, padx=100, pady=50)

        self._btn_feature_histogram = Button(preview_frame, width=26, text="Feature Histogram", command=lambda: self._feature_histogram(showing=1))
        self._btn_feature_histogram.grid(row=1, column=2, padx=100, pady=50)

        self._btn_correlation_heatmap = Button(preview_frame, width=26, text="Correlation Heatmap", command=lambda: self._correlation_heatmap(showing=1))
        self._btn_correlation_heatmap.grid(row=2, column=0, padx=100, pady=50)

        self._btn_PCA_2D_Scatter = Button(preview_frame, width=26, text="PCA 2D Scatter", command=lambda: self._PCA_2D_Scatter(showing=1))
        self._btn_PCA_2D_Scatter.grid(row=2, column=1, padx=100, pady=50)

        self._btn_PCA_3D_Scatter = Button(preview_frame, width=26, text="PCA 3D scatter", command=lambda: self._PCA_3D_Scatter(showing=1))
        self._btn_PCA_3D_Scatter.grid(row=2, column=2, padx=100, pady=50)

        ml_frame = Frame(self.master)
        ml_frame.grid(row=3, column=0, sticky=N, pady=100)

        self._label_ml = Label(ml_frame, text="Machine Learning", fg='blue', font=("Helvetica", 16))
        self._label_ml.grid(row=0, column=0, columnspan=3)

        self._btn_Logistic_Regression = Button(ml_frame, width=26, text="Logistic Regression", command=lambda: self._logisti_regression())
        self._btn_Logistic_Regression.grid(row=1, column=0, padx=100, pady=50)

        self._btn_SVM = Button(ml_frame, width=26, text="SVM", command=lambda: self._svm())
        self._btn_SVM.grid(row=1, column=1, padx=100, pady=50)

        self._btn_KNN = Button(ml_frame, width=26, text="KNN", command=lambda: self._knn())
        self._btn_KNN.grid(row=1, column=2, padx=100, pady=50)

        self._btn_RandomForest = Button(ml_frame, width=26, text="RandomForest", command=lambda: self._random_forest())
        self._btn_RandomForest.grid(row=2, column=0, padx=100, pady=50)

        self._btn_K_Means = Button(ml_frame, width=26, text="K-means", command=lambda: self._kmeans())
        self._btn_K_Means.grid(row=2, column=1, padx=100, pady=50)

        self._btn_Bayes = Button(ml_frame, width=26, text="Bayes", command=lambda: self._naive_bayes())
        self._btn_Bayes.grid(row=2, column=2, padx=100, pady=50)

        process_frame = Frame(self.master)
        process_frame.grid(row=0, column=1, rowspan=4, sticky=W+E+N+S)
        self._text_process = Text(process_frame, state=DISABLED)
        self._text_process.pack(fill=BOTH, expand=True)

    def _open_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[('Excel Files', ' .csv .xls .xlsx')])
        self._text_filename.config(state=NORMAL)
        self._text_filename.insert(END, self.filename)
        self._text_filename.config(state=DISABLED)

    def _submit_data(self):
        assert self.filename != "", self._send_text("No file selected")
        self.submit = True
        self._send_text("Reading the file, please wait...")
        self._send_text(self.all_data.read_data_from_path(self.filename))

    def _basic_feature(self, showing=0):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        if self.v.get() == "plotly":
            res = self.all_data.basic_feature_table()
        if self.v.get() == "matplotly":
            res = self.all_data.basic_feature_table_mat()

        self._send_text(res[1])
        if showing:
            res[0].show()

        return res[0]

    def _label_pie(self, showing=0):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        if self.v.get() == "plotly":
            res = self.all_data.label_pie_chart()
        if self.v.get() == "matplotly":
            res = self.all_data.label_pie_chart_mat()

        self._send_text(res[1])
        if showing:
            res[0].show()
        return res[0]

    def _feature_histogram(self, showing=0):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        if self.v.get() == "plotly":
            res = self.all_data.label_feature_histogram()
        if self.v.get() == "matplotly":
            res = self.all_data.label_feature_histogram_mat()

        self._send_text(res[1])
        if showing:
            res[0].show()
        return res[0]

    def _PCA_2D_Scatter(self, showing=0):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        if self.v.get() == "plotly":
            res = self.all_data.PCA_2D_Scatter()
        if self.v.get() == "matplotly":
            res = self.all_data.PCA_2D_Scatter_mat()

        self._send_text(res[1])
        if showing:
            res[0].show()
        return res[0]

    def _PCA_3D_Scatter(self, showing=0):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        if self.v.get() == "plotly":
            res = self.all_data.PCA_3D_Scatter()
        if self.v.get() == "matplotly":
            res = self.all_data.PCA_3D_Scatter_mat()

        self._send_text(res[1])
        if showing:
            res[0].show()
        return res[0]

    def _correlation_heatmap(self, showing=0):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        if self.v.get() == "plotly":
            res = self.all_data.correlation_heatmap()
        if self.v.get() == "matplotly":
            res = self.all_data.correlation_heatmap_mat()

        self._send_text(res[1])
        if showing:
            res[0].show()
        return res[0]

    def _logisti_regression(self):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        self._send_text("Logistic Regression processing...")
        self.ml.set_data(feature_data=self.all_data.standard_feature_data, label=self.all_data.label)
        res = self.ml.logistic_regression()
        self._send_text("Logistic Regression result:")
        for _, val in res.items():
            self._send_text(val)

    def _svm(self):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        self._send_text("SVM processing...")
        self.ml.set_data(feature_data=self.all_data.standard_feature_data, label=self.all_data.label)
        res = self.ml.SVM()
        self._send_text("SVM result:")
        for _, val in res.items():
            self._send_text(val)

    def _knn(self):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        self._send_text("KNN processing...")
        self.ml.set_data(feature_data=self.all_data.standard_feature_data, label=self.all_data.label)
        res = self.ml.KNN()
        self._send_text("KNN result:")
        for _, val in res.items():
            self._send_text(val)

    def _random_forest(self):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        self._send_text("Random Forest processing...")
        self.ml.set_data(feature_data=self.all_data.feature_data, label=self.all_data.label)
        res = self.ml.RandomForest()
        self._send_text("Random Forest result:")
        for _, val in res.items():
            self._send_text(val)

    def _kmeans(self):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        self._send_text("K-Means processing...")
        self.ml.set_data(feature_data=self.all_data.standard_feature_data, label=self.all_data.label)
        res = self.ml.K_Means()
        self._send_text("K-Means result:")
        for _, val in res.items():
            self._send_text(val)

    def _naive_bayes(self):
        assert self.filename != "", self._send_text("No file selected")
        assert self.submit, self._send_text("Please click submit button")
        self._send_text("Naive Bayes processing...")
        self.ml.set_data(feature_data=self.all_data.feature_data, label=self.all_data.label)
        res = self.ml.Naive_Bayes()
        self._send_text("Naive Bayes result:")
        for _, val in res.items():
            self._send_text(val)



    def _send_text(self, content):
        self._text_process.config(state=NORMAL)
        self._text_process.insert(END, content + '\n')
        self._text_process.config(state=DISABLED)


if 'win' in sys.platform:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
root = Tk()
root.geometry("2600x1300")
# root.tk.call('tk', 'scaling', 1.0)
# root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(3, weight=1)
# root.resizable(False, False)
analysis = Interface(master=root)
root.mainloop()