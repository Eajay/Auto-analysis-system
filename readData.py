import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn import decomposition
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


color = ["rgba(152, 0, 0, .8)", "rgba(29, 240, 247, .8)", "rgba(100, 232, 59, .8)", "rgba(224, 242, 59, .8)"]


class DataPreview:
    def __init__(self):
        self.file_types = [".csv", ".xls", "xlsx"]

    def read_data_from_path(self, path):
        res = ""
        assert (path[-4:] in self.file_types or path[-5:] in self.file_types), "File type not supported."
        try:
            self.raw_pd_data = pd.read_csv(path)
            res = "Read finished, supporting .csv file"
        except:
            self.raw_pd_data = pd.read_excel(path)
            res = "Read finished, supporting .xls or .xlsx file"

        self.raw_pd_without_missing_data = self.raw_pd_data.dropna(how='any')

        self.column_name = list(self.raw_pd_data.columns)

        self.raw_data = self.raw_pd_data.values

        self.raw_without_missing_data = self.raw_pd_without_missing_data.values

        self.feature_data = self.raw_without_missing_data[:, :-1]

        self.label = self.raw_without_missing_data[:, -1].astype(int)

        self.standard_feature_data = preprocessing.StandardScaler().fit_transform(self.feature_data)

        return res


    @property
    def valid_data_number(self):
        return self.raw_without_missing_data.shape[0]

    @property
    def invalid_data_number(self):
        return self.raw_data.shape[0] - self.raw_without_missing_data.shape[0]

    def basic_feature_table_mat(self):
        column = self.column_name[:-1]
        row = ["maximum value", "minimum value", "average value", "standard deviation"]

        res = []
        for val in self.feature_data.T:
            tmp = []
            tmp.append(np.max(val))
            tmp.append(np.min(val))
            tmp.append(np.average(val))
            tmp.append(np.std(val))
            res.append(tmp)

        res = np.around(np.array(res).T, decimals=3)

        fig = plt.figure(figsize=[36, 16])
        ax = fig.add_subplot(111)

        the_table = ax.table(cellText=res,
                             rowLabels=row,
                             colLabels=column,
                             cellLoc='center',
                             loc='center',)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(15)
        the_table.scale(1, 4)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        plt.show()
        return plt, "Showing Basic Feature Table"

    def basic_feature_table(self):
        """
        :return: maximum, minimum, average and standard deviation
        of each feature in plotly table
        """
        # return pandas dataframe
        column = self.column_name[:-1]
        row = ["maximum value", "minimum value", "average value", "standard deviation"]

        res = []
        for val in self.feature_data.T:
            tmp = []
            tmp.append(np.max(val))
            tmp.append(np.min(val))
            tmp.append(np.average(val))
            tmp.append(np.std(val))
            res.append(tmp)

        # self.df = pd.DataFrame(np.array(res).T, index=row, columns=column)
        self.df = pd.DataFrame(np.array(res).T, columns=column)
        self.df.insert(0, "feature number", row, True)
        fig = go.Figure(data=go.Table(
            header=dict(values=list(self.df.columns),
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(color='black', size=18)),
            cells=dict(values=self.df.values.T,
                       fill_color='lavender',
                       align='left',
                       font=dict(size=16),
                       height=30)
        ))
        return fig, "Showing Basic Feature Table"

    def label_pie_chart_mat(self):
        labels, count = np.unique(self.label, return_counts=True)

        explode = [0] * len(labels)
        explode[np.argmax(count)] = 0.1

        fig = plt.figure(figsize=[15, 12])
        ax = fig.add_subplot(111)
        ax.pie(count, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)

        # plt.show()
        return plt, "Showing Label Pie Chart"

    def label_pie_chart(self):
        """

        :return: plotly pie chart of label distribution
        """
        labels, count = np.unique(self.label, return_counts=True)
        # fig = go.Figure(data=[go.Pie(labels=labels, values=count)],
        #                 layout=go.Layout(title=go.layout.Title(text="Label Pie Chart", title_x=0.5))
        #                 )
        fig = go.Figure(data=[go.Pie(labels=labels, values=count)])
        fig.update_layout(title_text="Label Pie Chart", title_x=0.5)
        return fig, "Showing Label Pie Chart"

    def label_feature_histogram_mat(self):
        fig1 = plt.figure(figsize=[20, 16])
        ax1 = fig1.add_subplot(111)
        self.raw_pd_without_missing_data.hist(ax=ax1, bins=10)
        # plt.show()
        return plt, "Showing Histogram"

    def label_feature_histogram(self):
        """

        :return: plotly histogram dictionary based on feature name
        """
        column_num = 2

        fig = make_subplots(rows=int(len(self.column_name[: -1]) / column_num) + 1, cols=column_num)
        for i, val in enumerate(self.column_name[: -1]):
            fig.add_trace(go.Histogram(x=self.feature_data[:, i], name=val),
                          row=int(i / column_num) + 1, col=int(i % column_num) + 1,)

        # fig.update_layout(barmode='stack')
        # fig.show()
        return fig, "Showing Histogram"

        # fig_dic = {}
        # for i, val in enumerate(self.column_name[:-1]):
        #     tmp = px.histogram(self.raw_pd_without_missing_data, x=val,
        #                        title="Histogram of " + val, color=self.column_name[-1],
        #                        marginal='rug', hover_data=self.raw_pd_without_missing_data.columns
        #                        )
        #     fig_dic[val] = tmp
        # return fig_dic


    def PCA_2D_Scatter_mat(self):
        pca_2d = decomposition.PCA(n_components=2)
        data = pca_2d.fit_transform(self.standard_feature_data)

        # create label dictionary
        label_dic = {}

        for training_data, target in zip(data, self.label):
            if target not in label_dic:
                label_dic[target] = [training_data]
            else:
                label_dic[target].append(training_data)

        fig = plt.figure(figsize=[20, 16])
        ax = fig.add_subplot(111)
        color = ['red', 'blue', 'yellow', 'green', 'black']
        color_index = 0
        for key, val in label_dic.items():
            ax.scatter(np.array(val)[:, 0], np.array(val)[:, 1], alpha=0.7, c=color[color_index], s=35, label=key)
            color_index += 1
        ax.legend(loc="upper right", prop={'size': 20})
        # plt.show()
        return plt, "Showing PCA 2D scatter"

    def PCA_2D_Scatter(self):
        """
        transform data into 2D by PCA
        :return: 2D scatter figure
        """
        pca_2d = decomposition.PCA(n_components=2)
        data = pca_2d.fit_transform(self.standard_feature_data)

        # create label dictionary
        label_dic = {}

        for training_data, target in zip(data, self.label):
            if target not in label_dic:
                label_dic[target] = [training_data]
            else:
                label_dic[target].append(training_data)

        fig = go.Figure()
        for i, key in enumerate(label_dic.keys()):
            tmp = np.array(label_dic[key])
            fig.add_trace(go.Scatter(
                x=tmp[:, 0], y=tmp[:, 1],
                name=str(key),
                mode='markers',
                marker_color=color[i]
            ))

        fig.update_traces(mode='markers', marker_line_width=2, marker_size=12)
        fig.update_layout(title_text='PCA 2D Scatter', title_x=0.5, yaxis_zeroline=False, xaxis_zeroline=False)
        return fig, "Showing PCA 2D scatter"

    def PCA_3D_Scatter_mat(self):
        pca_3d = decomposition.PCA(n_components=3)
        data = pca_3d.fit_transform(self.standard_feature_data)

        # create label dictionary
        label_dic = {}

        for training_data, target in zip(data, self.label):
            if target not in label_dic:
                label_dic[target] = [training_data]
            else:
                label_dic[target].append(training_data)

        fig = plt.figure(figsize=[20, 16])
        ax = fig.add_subplot(111, projection='3d')
        color = ['red', 'blue', 'yellow', 'green', 'black']
        color_index = 0
        for key, val in label_dic.items():
            ax.scatter(np.array(val)[:, 0], np.array(val)[:, 1], np.array(val)[:, 2], alpha=0.7, c=color[color_index], s=50, label=key)
            color_index += 1
        ax.legend(loc="upper right", prop={'size': 20})
        # plt.show()
        return plt, "Showing PCA 3D scatter"

    def PCA_3D_Scatter(self):
        """
        transform data to 3D by PCA
        :return: 3D scatter figure
        """
        pca_3d = decomposition.PCA(n_components=3)
        data = pca_3d.fit_transform(self.standard_feature_data)

        # create label dictionary
        label_dic = {}

        for training_data, target in zip(data, self.label):
            if target not in label_dic:
                label_dic[target] = [training_data]
            else:
                label_dic[target].append(training_data)

        fig = go.Figure()
        for i, key in enumerate(label_dic.keys()):
            tmp = np.array(label_dic[key])
            fig.add_trace(go.Scatter3d(
                x=tmp[:, 0], y=tmp[:, 1], z=tmp[:, 2],
                name=str(key),
                mode='markers',
                marker_color=color[i]
            ))

        fig.update_traces(mode='markers', marker_line_width=0.2, marker_size=6)
        fig.update_layout(title_text='PCA 3D Scatter', title_x=0.5, yaxis_zeroline=False, xaxis_zeroline=False)
        return fig, "Showing PCA 3D scatter"

    def correlation_heatmap_mat(self):
        correlation_matrix = self.raw_pd_without_missing_data.corr()
        fig = plt.figure(figsize=[26, 18])
        ax = fig.add_subplot(111)
        sns.set(font_scale=2)
        sns.heatmap(correlation_matrix, annot=True, linewidths=.2, ax=ax, cmap='Blues', annot_kws={'size': 16})
        ax.tick_params(labelsize=15)
        # plt.show()
        return plt, "Showing Correlation Heatmap"

    def correlation_heatmap(self):
        correlation_matrix = self.raw_pd_without_missing_data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=self.column_name,
            y=self.column_name,
            colorscale='blues',
            # burg
            colorbar=dict(
                title="correlation number",
                titleside="top",
                titlefont=dict(size=16),
                tickmode="array",
                ticks="outside"
            )
        ))
        fig.update_layout(title_text="Correlation Heatmap", title_x=0.5)
        return fig, "Showing Correlation Heatmap"





# data = DataPreview()
# data.read_data_from_path(path="diabetes.csv")
# data.correlation_heatmap_mat()
# data.PCA_3D_Scatter_mat()
# data.label_feature_histogram_mat()
# data.basic_feature_table_mat()
# data.label_pie_chart_mat()
# data.label_feature_histogram()
# data.correlation_heatmap()
# data.PCA_2D_Scatter()
# data.PCA_3D_Scatter()
# data.label_pie_chart()
# data.basic_feature_table()
#
    # res = data.label_feature_histogram()
    # for key, value in res.items():
    #     value.show()
# print(data.label_pie_chart())
# res = data.label_feature_histogram()
# res['a'].show()
# res['b'].show()
# df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
# print(df.head(6))
# print(data.basic_feature_table().head(5))
# data.label_pie_chart().show()
# res = data.label_feature_histogram()
# res[data.column_name[1]].show()

