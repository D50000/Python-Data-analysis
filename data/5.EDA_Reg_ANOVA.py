
# coding: utf-8

# # 探索式資料分析與線性模型
# 
# ## 大綱
# 
# ### A. 探索式資料分析與常用繪圖語法(以NBA 2013-2014賽季的球員數據為例)
# ### B. 線性迴歸
# ### C. 變異數分析

### A. 探索式資料分析與常用繪圖語法(以NBA 2013-2014賽季的球員數據為例)
# ## NBA常見關鍵字

# ### 隊伍

# - 大西洋組
#     - 波士頓塞爾蒂克	BOS	Boston Celtics
#     - 新澤西籃網	NJ	New Jersey Nets
#     - 紐約尼克	NY	New York Knicks
#     - 費城76人	PHI	Philadelphia 76ers
#     - 多倫多暴龍	TOR	Toronto Raptors
# - 中央組
#     - 芝加哥公牛	CHI	Chicago Bulls
#     - 克里夫蘭騎士	CLE	Cleveland Cavaliers
#     - 底特律活塞	DET	Detroit Pistons
#     - 印地安那溜馬	IND	Indiana Pacers
#     - 密爾瓦基公鹿	MIL	Milwaukee Bucks
# - 東南組		
#     - 亞特蘭大老鷹	ATL	Atlanta Hawks
#     - 夏洛特山貓	CHA	Charlotte Bobcats
#     - 邁阿密熱火	MIA	Miami Heat
#     - 奧蘭多魔術	ORL	Orlando Magic
#     - 華盛頓巫師	WAS	Washington Wizards	
# - 西南組
#     - 達拉斯小牛	DAL	Dallas Mavericks
#     - 休士頓火箭	HOU	Houston Rockets
#     - 孟斐斯灰熊	MEM	Memphis Grizzlies
#     - 紐奧良黃蜂	NO	New Orleans Hornets
#     - 聖安東尼奧馬刺	SAS	San Antonio Spurs
# - 西北組
#     - 丹佛金塊	DEN	Denver Nuggets
#     - 明尼蘇達灰狼	MIN	Minnesota Timberwolves
#     - 波特蘭拓荒者	POR	Portland Trail Blazers
#     - 西雅圖超音速	SEA	Seattle SuperSonics
#     - 猶他爵士	UTA	Utah Jazz
# - 太平洋組
#     - 金州勇士	GS	Golden State Warriors
#     - 洛杉磯快艇	LAC	Los Angeles Clippers
#     - 洛杉磯湖人	LAL	Los Angeles Lakers
#     - 鳳凰城太陽	PHO	Phoenix Suns
#     - 沙加緬度國王	SAC	Sacramento Kings

# ### POS先發位置

# - C即為中鋒，代表人物為歐肥
# - PF是大前鋒，代表人物有鄧肯，賈奈特，韋伯，馬丁
# - SF小前鋒，卡特，詹姆斯，史托傑維奇，路易斯
# - SG得分後衛，KOBE，T-MAC，皮爾斯，理察森，艾倫 
# - PG控球後衛，艾佛森，奈許，畢比，奇德

# ### 常用NBA數據縮寫
# ![](_img/nba-abb.png)

# ## 載入資料並理解資料

# ### 載入資料
import os
os.chdir('/Users/Vince/cstsouMac/Python/Slides/Basics/day1c')

import pandas as pd
nba = pd.read_csv("./data/nba_2013.csv")


# ### 列出球員數(row)及特徵數(col)


nba.shape


# ### 列出前五筆(row)資料



nba.head(5)


# ### 計算各特徵的平均值



nba.mean()


# ### 計算各欄位的敘述統計值
# - 並非每個欄位都是數值資料，pandas還是很聰明的只挑選了數值欄位
# - 警告出現是因為fg.及x3p.欄位中含有非數值資料，NaN
#     - 有些球員不只沒投過三分球，甚至沒投進過任何球，或者，資料遺失



nba.describe()


# ### 取得「fg.」為NaN的列資料


nba[nba['fg.'].isnull()]


# ### 練習
# 
# - 將「opendata103Y010.csv」資料讀入並進行摘要分析
# - 練習1：
#     - 載入資料
#     - 取得資料的維度狀況
#     - 預覽前5筆資料
#     - 計算各欄位摘要統計值
#     - 解答：`%pycat _homework/opendat1.py`
# - 練習2：
#     - 上面的練習面臨了什麼狀況呢？
#     - 請找出方法成功的計算，數值欄位的摘要統計值
#     - 語法上的不熟悉很正常，給自己3分鐘的時間Google
#     - 解答：`%pycat _homework/opendat2.py`

# ## 探索資料

# ### 最年輕的球員


nba.sort_values("age").head()


# ### 最年長的球員


nba.sort_values("age",ascending=False).head()


# ### 投籃命中最多次的球員


nba.sort_values("fg",ascending=False).head()


# ### 投籃命中率最高的球員



nba.sort_values("fg.",ascending=False).head()


# ### 抓到籃板球最多次的球員



nba.sort_values("trb",ascending=False).head()


# ### 助攻最多次的球員



nba.sort_values("ast",ascending=False).head()


# ### 得分最多的球員



nba.sort_values("pts",ascending=False).head()


# ### 練習
# 
# - 試著以上述方式，瞭解主要欄位的排名狀況
# 

# ### 年紀、助攻、命中次數、籃板球、得分之間的關係如何（成對散點圖）
# 
# 一個探索數據的常用方法是查看列與列之間有多相關。



get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns
sns.pairplot(nba[["age","ast", "fg", "trb","pts"]])


# ### 練習
# 
# - 任意挑選opendata103Y010資料的兩個數值欄位繪製成對散點圖
# - 解答：%pycat _homework/opendat3.py

# ## 將球員以k-means演算法集群(非監督式學習)
# 
# ### K-means背景知識
# - K-平均值演算法是最早出現的集群分析演算法之一，它是一種快速分群方法，但對於異常值或極值敏感，穩定性差，因此較適合處理分佈集中的大樣本資料集。
# - Select K points as the initial centroids(隨機選取K個初始群集中心點)
# - repeat Form K clusters by assigning all points to the closest centroid (指派所有點之歸群)，Recompute the centroid of each cluster (更新各群中心)
# - until The centroids don’t change
# - Limitations of K-means
# - Sizes (各群大小不同)
# - Densities (密度不同)
# - Non-globular shapes (非球形)
# - Outliers (離群值)
# - Gaming source http://etrex.blogspot.tw/2008/05/k-mean-clustering.html

# ### 使用年齡、命中次數、籃板球及得分進行集群(k=5)



from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = nba._get_numeric_data().dropna(axis=1) # 移除非數值及含有NA(NaN)值的欄位
kmeans_model.fit(nba[["age","fg","trb","pts"]])
labels = kmeans_model.labels_
print(labels)


# ### 第一群的前五位球員



nba.loc[labels == 1,["player","age","fg","trb","pts"]].head()


# ### 第二群的前五位球員



nba.loc[labels == 2,["player","age","fg","trb","pts"]].head()


# ### 第三群的前五位球員



nba.loc[labels == 3,["player","age","fg","trb","pts"]].head()


# ### 按類別繪製球員分佈圖
# - 首先使用PCA將資料降至2維，然後畫圖，用不同標記或深淺的點標誌類別。
# 
# <img src="_img/_pcavslda.png" alt="pca vs lda" style="width: 500px;"/>
# 



get_ipython().run_line_magic('pylab', 'inline')
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()


# ### 練習
# 
# - 練習1：
#     - 將opendata103Y010資料以k-means的**列**集為5群
#     - 依照上述步驟將其視覺化
#     - 提示：透過`apply(pd.to_numeric)`一次將多欄位轉換為數值資料
#     - 解答：`%pycat _homework/opendat4.py`
# - 練習2：
#     - 將labels對應地名(區域別+村里名稱)
#     - 解答：`%pycat _homework/opendat5.py`

### B. 線性迴歸
# # 線性迴歸(Linear Regression)
# 
# ||連續(continuous)|類別(categorical)|
# |---|---|---|
# |**監督式學習(supervised)**|**迴歸(regression)**|分類(classification)|
# |**非監督式學習(unsupervised)**|降維(dimension reduction)|集群(clustering)|
# 
# ## 為何學習線性迴歸？
# 
# - 使用範圍廣泛
# - 計算效率高
# - 使用難度低(沒有許多參數需調校)
# - 容易解釋
# - 為許多方法的基礎
# 
# ## 相關套件
# 
# - [Statsmodels](http://statsmodels.sourceforge.net/) 
# - [scikit-learn](http://scikit-learn.org/stable/)
# - [SciPy](https://www.scipy.org/)

# ## 透過虛擬案例學習

# ### 先有個假設
# 
# * 1.假設平均日休閒時間透過**神秘函數**能夠預測結婚的年齡：
#     - f(休閒時間) = 結婚年齡  
# * 2.假設輸入x,y兩個人的BMI值，透過**神秘函數**可以預測他們的速配指數：
#     - f(BMI_x, BMI_y) = x與y的速配指數

# ### 來學這個函數
# 
# 我們假設真實世界某個現象(例如上述假設1)的「理想函數」長這樣:
# 
# $$f(x) = -7.7x + 55$$

# ### 模擬 50 個學習數據
# 
# 就像真實的世界, 我們加入 noise。


x = np.linspace(0,5,50)
y = -7.7*x + 55 + randn(50)*10


# ### 看看我們的數據
# 
# 看看我們目前取的點, 附上「完全正確的」函數。


scatter(x,y)
plot(x, -7.7*x + 55,'b');


# ### 用 SciPy 來迴歸
# 
# - SciPy的好處是只要預想好這些資料長得很像什麼函數, 我們都可以迴歸! 不一定要線性函數!
# - 以下使用線性函數為例，先做一個目標函數
# 
# $$f(x) = ax + b$$
# 
# 再來算 a, b 這樣。


from scipy.optimize import curve_fit


def f(x, a, b):
    return a*x + b


# ### 開始迴歸
# 
# - 我們「學習的資料」就是剛剛的 x, y
#     - 若以一開始的假設來說，x就是某個人的平均日休閒時間，y就是他的實際結婚年齡
#     - 並且我們有50個實際案例(模擬的)
# - 傳回的 popt 是以最小平方法下得到的 a, b


popt, pcov = curve_fit(f, x, y)



pcov


popt


a, b = popt


# ### 畫出結果
# 
# 紅色是我們模擬出來的, 藍色是「理想狀況函數」。


scatter(x,y)
plot(x,-7.7*x + 55,'b',label="original")
plot(x, a*x + b, 'r',label="fit")
ax = gca()
ax.legend();


# <font size=6 style="text-shadow:0px 0px 15px #FF37FD;">練習</font>
# - 將「alligator.csv」資料讀入並進行線性迴歸分析
# - 練習1：
#     - 載入資料
#     - 取得資料的維度狀況
#     - 試以長度為自變數(x)，重量為應變數(y)，配適迴歸模型
#     - 繪製散點圖並將迴歸直線一同畫上
#     - 解答：`%pycat _homework/reg1.py`

# ## 案例實作：廣告資料
# 
# 試著理解資料、提出問題並且透過迴歸模型去回答問題


# 套件載入
import pandas as pd


# 讀取資料為DataFrame物件
data = pd.read_csv('_data/Advertising.csv', index_col=0)
data.head()


# **特徵(features)，自變數**：
# - TV：某產品投入於電視廣告的金額
# - Radio：某產品投入於廣播廣告的金額
# - Newspaper：某產品投入於報紙廣告的金額
# 
# **反應(response)，應變數**：
# - Sales: 於給定的市場範圍中，某產品的銷售狀況


# print the shape of the DataFrame
data.shape


# - 200個**觀測值(observations)**
# - 4個**變數(variable)**


# 以散點圖呈現特徵與回應之間的關係
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])


# ### Questions About the Advertising Data
# 
# 想像你在這間生產並營銷此產品的公司上班，公司可能問你：如何透過這份資料去規劃廣告預算？
# 
# 更具體的來說，可以列出以下問題：
# 1. 廣告與銷售之間有關係嗎？
# 2. 關係的強度？
# 3. 哪種廣告類型比較有貢獻？
# 4. 每種廣告類型對於銷售的影響如何？
# 5. 可以依據投入的廣告預算，預測銷售狀況嗎？
# 

# ### 簡單線性迴歸(Simple Linear Regression)
# 
# 透過單一特徵(**single feature**)或稱自變數(independent variable)、預測變數(predictor)，預測一個**值**(**quantitative response**)或稱依變數(dependent variable)、反應變數(response variable)。公式如下：
# 
# $y = \beta_0 + \beta_1x$
# 
# 定義如下：
# - $y$ is the response
# - $x$ is the feature
# - $\beta_0$ is the intercept(截距)
# - $\beta_1$ is the coefficient(係數) for x
# - $\beta_0$與$\beta_1$稱**模型係數(model coefficients)**
# 
# 為了建立(配適)此迴歸模型，我們必須把模型係數**"學"**出來，學成後就可以預測銷售了！

# ### 如何估計("學") 模型係數？
# 
# 常見的方法為**最小平方法(least squares criterion)**，其代表的意義是要找到一條**殘差總和(sum of squared residuals)**最小的**線**，殘差也就是誤差(error):

# <img src="_img/estimating_coefficients.png">

# 左圖解釋如下：
# - 黑點表示**觀測值(observed values)**，一個x(自變數，投入廣告成本)與一個y(應變數，銷售狀況)
# - 藍線表示**最小平方線(least squares line)**
# - 紅線表示**殘差(residuals)**
# 
# 模型係數與最小平方線的關係為何？
# - $\beta_0$為**截距(intercept)** (當$x$=0時的$y$)
# - $\beta_1$為**斜率(slope)** (當$x$變動時，$y$改變的量)

# <img src="_img/slope_intercept.png">

# 以下透過 **Statsmodels** 套件來估計此廣告資料的模型係數：


# this is the standard import if you're using "formula notation" 
import statsmodels.formula.api as smf

# create a fitted model in one line
# 這邊以Sales為y也就是response，TV為x也就是feature
# ordinary least squares
lm = smf.ols(formula='Sales ~ TV', data=data).fit()

# print the coefficients
lm.params


# ### 模型係數解釋
# 
# 如何解釋TV係數($\beta_1$)?
# - 每增加一個單位的TV廣告成本相當於增加0.047537個"unit"的銷售量
# - 白話來說，每花1000元廣告費會多銷售47.537個產品
# 
# 注意：若$\beta_1$為負數，表示增加廣告費用，反而會降低銷售量

# ### 使用此模型進行預測
# 
# 當模型係數已經**"學"**成，我們就可以透過新的廣告預算進行銷售預測，假設我們投入的TV廣告預算為**$50,000**。
# 
# $$y = \beta_0 + \beta_1x$$
# $$y = 7.032594 + 0.047537 \times 50$$


# manually calculate the prediction
7.032594 + 0.047537*50


# 預測的銷售量為**9,409**個產品
# 
# 當然，我們也可以透過Statsmodels的函數幫助我們更方便地進行預測：


# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'TV': [50]})
X_new.head()


# use the model to make predictions on a new value
lm.predict(X_new)


# ### 繪製最小平方線
# 
# 已經有了二元一次方程式，透過兩點一線的概念即可繪製該直線，這邊我們透過TV觀測值中的最小值及最大值及它們對應的預測值來繪製：


# create a DataFrame with the minimum and maximum values of TV
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()



# make predictions for those x values and store them
preds = lm.predict(X_new)
preds



# first, plot the observed data
data.plot(kind='scatter', x='TV', y='Sales')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)


# <font size=6 style="text-shadow:0px 0px 15px #FF37FD;">練習</font>
# - 以Radio為自變數進行單變量線性迴歸
# - 假設一個新的Radio投入成本為30，請預測其銷售量
# - 依上述方式進行視覺化
# - 解答：`%pycat _homework/reg1.py`

# ### 模型的可信程度？
# 
# - 相關的概念為**信賴區間(confidence intervals)**
# - Statsmodels預設的信賴區間為95%，這表示：
#     - If the population(母體) from which this sample(抽樣) was drawn was **sampled 100 times**, approximately **95 of those confidence intervals** would contain the "true"(真正?) coefficient.



# print the confidence intervals for the model coefficients
# 係數的區間
lm.conf_int()


# - 很難有辦法取得整個母體的資料，通常是以樣本資料進行係數估計
# - "真正的"係數是否在我們估計的區間之中呢？there's no way to actually know
# - 我們透過僅有的資料進行估計，並且用一個範圍來表示這個"真正的"係數**可能**在其中
# - 95%的信賴區間只是一個慣例，實際上可以依照我們的需求去決定，要了解的是：
#     - 信賴區間定的越小(假設90%)，係數的範圍就會越小
#     - 信賴區間定的越大(假設99%)，係數的範圍就會越大
#     - 就好像我說：
#         - 我今晚打麻將有20%的信心能贏1000-1200元之間(係數範圍小)
#         - 我今晚打麻將有99%的信心，輸贏會介於賠1000跟賺2000元之間(係數範圍大)
# 

# ### 假設檢定及p值(p-values)
# 
# 與信賴區間密切相關的是**假設檢定(hypothesis testing)**。廣義來說，我們會假設一個**虛無假設(null hypothesis)**與一個跟虛無假設相反的**對立假設(alternative hypothesis)**。接著，透過手上的資料去評估，**拒絕虛無假設(rejecting the null hypothesis)**或者是**不拒絕虛無假設(failing to reject the null hypothesis)**.
# 
# (Note that "failing to reject" the null is not the same as "accepting"(我不拒絕你不表示接受你...) the null hypothesis. The alternative hypothesis may indeed be true, except that you just don't have enough data to show that.)
# 
# 用於模型係數上，慣例使用下列假設檢定：
# - **虛無假設:** 電視廣告與銷售不相關( $\beta_1 = 0$)
# - **對立假設:** 電視廣告與銷售相關 ( $\beta_1 \neq 0$)
# 
# 如何檢定呢？
# - 就是看95%的信賴區間中，是否不包含0
# - **p值(p-value)**表示，該係數為0的機率
# 


# print the p-values for the model coefficients
lm.pvalues


# - 若95%的信賴區間**包含0**，則p值將會大於0.05，反之，小於0.05。
# - 再次提醒，0.05只是一個慣例
# - 通常會忽略截距的p值
# - 以此案例來說，p值遠小於0.05，所以我們**相信**電視廣告與銷售量是有關係的。
# 

# ### 模型的配適程度如何？
# 
# - 普遍使用的方法是透過**決定係數(R-squared)**或稱R平方、迴歸可解釋變異量比
# - 決定係數是指此迴歸模型可以解釋這些樣本資料的變異量比例
# - 決定係數是介於0到1之間，越大越好，表示越多的變異量被此模型解釋
# - 用下圖可以更好理解

# <img src="_img/r_squared.png">

# - 藍線：解釋了部分的變異量，其決定係數為0.54
# - 綠線：解釋了更多的變異量，其決定係數為0.64
# - 紅線：解釋了的變異量更多且更深入，其決定係數為0.66(是否可能過度配適?)
#     - ![](_img/overfitting.png)
# 
# 以下計算方才模型的決定係數：


# print the R-squared value for the model
lm.rsquared


# - 這樣的決定係數是否好呢？其實很難說！
# - 一個好的決定係數，取決於應用情境，例如：
#     - 儀器的精確度校正，通常決定係數要高達0.999
#     - 社會科學的研究，通常決定係數在0.6-0.7
# - 所以，決定係數更好的用法是在於**比較不同的模型**時

# ### 多變量迴歸
# 
# - 簡單線性迴歸可以簡單的擴展為多特徵(自變量)
# - $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$
# - 每個$x$表示不同的特徵且各有其係數：
#     - $y = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper$


# create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the coefficients
lm.params


# 解釋：
# 
# - 在給定的Radio及Newspaper廣告投入成本下，每增加1000元的TV廣告，可以提高45.765單位的銷售量
# - 透過summary可以取得更多的模型資訊：


# print a summary of the fitted model
lm.summary()


# 以下就一些比較關鍵的部分說明：
# 
# - TV及Radio有顯著的**p值**，反觀Newspaper沒有。因此我們拒絕TV及Radio的虛無假設，不拒絕Newspaper的虛無假設。
# - TV及Radio的廣告投入成本與銷售量為正相關，而Newspaper的部分為(微微的)負相關，不過那已經不要緊，我們已經"不拒絕其虛無假設"
# - 此模型的決定係數高達0.897，比前一個模型還高，這表示三個特徵納入時，模型比只考慮TV時有更好的配適

# ### 特徵挑選(Feature Selection)
# 
# 如何決定哪些要挑選哪些特徵來進行線性模型配適呢？：
# - 都試試看，並且保留p值比較小的模型
# - 檢查每次新增新的特徵值，是否決策係數將會提高
# 
# 此方法的缺點為何？
# - 線性模型依賴於許多假設，例如特徵之間要各自獨立，若是違反了這些假設決定係數及p值就會降低可性度
# - 決定係數容易造成**過度配適(overfitting)**，因此沒辦法保證模型的決定係數越高一定越好。以下為例：
# 


# only include TV and Radio in the model
lm = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
lm.rsquared


# add Newspaper to the model (which we believe has no association with Sales)
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lm.rsquared


# **R-squared will always increase as you add more features to the model**, even if they are unrelated to the response. Thus, selecting the model with the highest R-squared is not a reliable approach for choosing the best linear model.
# 
# 更好的特徵挑選方法，**交叉驗證(Cross-validation, CV)**，能夠更好的估計樣本外誤差(out-of-sample error)，scikit-learn套件有相關方法，可以善用其達到更好的特徵挑選及參數調校，更重要的是CV能夠被應用在任何模型。
# 
# #### 秒懂交叉驗證
# ![](_img/10fold-cv.png)

# ### Linear Regression in scikit-learn
# 
# Let's redo some of the Statsmodels code above in scikit-learn:


# create x and y
feature_cols = ['TV', 'Radio', 'Newspaper']
x = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


# pair the feature names with the coefficients
list(zip(feature_cols, lm.coef_))


# predict for a new observation
lm.predict(pd.Series([100, 25, 25]).reshape(1, -1))


# calculate the R-squared
lm.score(x, y)


# 比較可惜的是**p-values**及**confidence intervals**在scikit-learn上取得有些不方便
# 
# 有興趣的請參考[HERE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)

# ## 再次以NBA資料為例(sklearn套件)
# ### 將資料劃分為訓練集與測試集
# - 將資料劃分為訓練集和測試集是避免過擬合的辦法之一


train = nba.sample(frac=0.8, random_state=1)
test = nba.loc[~nba.index.isin(train.index)]


# ### 假設我們希望通過球員的投籃命中預測其助攻次數


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[["fg"]], train["ast"])
predictions = lr.predict(test[["fg"]])


# ### 將原始值與預估值以DataFrame共同列出
# - test[['ast']]為DataFrame
# - test['ast']為Series
# - test['ast'].values為ndarray


print(type(test[['ast']]))
print(type(test['ast']))
print(type(test['ast'].values))


# ### 以Series合併的話，可能會有index的錯亂，故以ndarray合併


test['ast'].head()


pd.DataFrame({'ast':test['ast'].values,'pred':predictions}).head(10)


# ### 計算此模型的統計摘要


import statsmodels.formula.api as sm
model = sm.ols(formula='ast ~ fga', data=train)
fitted = model.fit()
fitted.summary()


# ### 計算誤差，這邊使用MSE(Mean Squared Error)


from sklearn.metrics import mean_squared_error
mean_squared_error(test["ast"], predictions)
      

### C. 變異數分析
# # ANOVA 變異數分析 (Analysis of Variance) - 1
# 
# 雖然叫做變異數分析
# 
# 但他比較的仍然是組與組之間對於某一個依變數的**「平均數」**是否有顯著的差異
# 
# 那到底要怎樣才算是有「顯著差異」？
# 
# 實作[Belleaya 部落格](http://belleaya.pixnet.net/blog/post/30754486)的有趣範例

# ### 這兩組的曲線長這樣子
# ![](_img/anova_1.png)
# 
# ### 所有樣本的平均值(總平均)大約在中間
# ![](_img/anova_2.png)
# 
# ### 假想我們在第2組挑一個樣本
# ![](_img/anova_3.png)
# 
# ### 組內、組間及總變異示意圖
# ![](_img/anova_4.png)
# 
# ## 最後的關鍵是F值
# 
# - F值的大小，決定了組跟組之間是否有顯著的差異
# - F值越大代表：
#     - 組跟組之間差異越大
#     - 組內差異越小
# - 針對自由度來查表，看看F值是否大到超過顯著的臨界值，則代表組與組之間有顯著的差異


import pandas as pd
datafile="_data/noodle.csv"
data_noodle = pd.read_csv(datafile)
 
#Create a boxplot
data_noodle.boxplot('score', by='group', figsize=(8, 5));


# ## ANOVA表格
# 
# 基本上是先看後面顯著性(PR, p值)這邊很顯然是非常顯著(比.001還要小)
# 
# > p值定義：實際上沒有差異，卻因誤差或偶發而產生資料差距（嚴格來說，更極端的差距也包含在內）的機率，依慣例要在5%以下。

import statsmodels.api as sm
from statsmodels.formula.api import ols
 # ordinary least squares
mod = ols('score ~ group',
                data=data_noodle).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


SS_B = aov_table['df'][0].astype(np.int64)
SS_W = aov_table['df'][1].astype(np.int64)
F = aov_table['F'][0]

print('分析結果顯示為顯著(F(',
      SS_B,
      ',',
      SS_W,
      ',)=',
      F,
      '，p<.001)，故拒絕虛無假設，即兩組麵食的得分具有顯著差異，豚骨拉麵得分顯著高於奶油烏龍麵。')


# ## ANOVA -2
# - 參考：https://youtu.be/DW0TnDYb82M?t=18m11s
# - 我們想了解學生對不同教學法喜愛程度是否有差異，A組以PPT教學，B組直接黑板演練，C組使用線上教學，隨機抽取三種教學法學生各5個人，以1-10的分數請他們評分如下：


score = [[8,4,5,5,4], # A組
         [8,5,9,7,9], # B組
         [4,4,6,6,5]]  # C組

### ANOVA法一
group_mean = [sum(i)/len(i) for i in score]
total_mean = sum(group_mean)/len(group_mean) # len(score) changed to len(group_mean) for better understanding
print("組平均： ", group_mean)
print("總平均： ", total_mean)


import pandas as pd
group_mean = pd.Series(group_mean)


# ![image.png](_img/anova-table.png)



SSB = sum(5*((group_mean-total_mean)**2)) #組間變異，自由度 （3-1=2)
SSW = sum([sum((pd.Series(score[i])-group_mean[i])**2) for i in range(3)]) #組內變異，自由度15-3=12
SST = SSB + SSW
MSB = SSB/2
MSW = SSW/12


print("SSB:" ,SSB, ", 自由度:", 2, ", MSB:" , MSB)
print("SSE:" ,SSW, ", 自由度:", 12, ", MSW:" ,MSW)
print("F:", MSB/MSW)



from scipy.stats import f
f.sf(MSB/MSW, 2, 12) # p-value from the survival function (sf)

### ANOVA法二
# ### 使用statsmodel計算前，先進行資料變形


score_df = pd.DataFrame(score).transpose()
score_df.columns = ["A","B","C"]
score_df = score_df.melt()
score_df


get_ipython().run_line_magic('matplotlib', 'inline')


# ### 使用boxplot感受一下組與組之間的變異

score_df.columns
score_df.boxplot('value', by='variable', figsize=(10, 7));



import statsmodels.api as sm
from statsmodels.formula.api import ols
 
mod = ols('value ~ variable',
                data=score_df).fit()
                
aov_table = sm.stats.anova_lm(mod, type=2)
print(aov_table)

### D. __name__的意義
# # 載入另一個python檔

# - 建立一個檔，名為toCall，待會「進行呼叫」
# - magic function 幫我們做到文件編寫的功能


get_ipython().run_cell_magic('writefile', 'Basics/data/toCall.py', 'import beCall')


# - 建立另一個檔，名為beCall，待會「被呼叫」


get_ipython().run_cell_magic('writefile', 'Basics/data/beCall.py', "print('__name__:' + __name__)")


# - 試試直接執行beCall.py
# - 此時的「`__name__`」會等於「`__main__`」



get_ipython().system('python "Basics/data/beCall.py"')
# __name__:__main__

# - 試試直接執行toCall.py
# - toCall會用import去載入beCall module
# - 此時的「`__name__`」會等於「`__beCall__`」



get_ipython().system('python "Basics/data/toCall.py"')
# AttributeError: 'NoneType' object has no attribute 'name'
]

# ### `__name__`代表什麼？



get_ipython().run_cell_magic('writefile', 'Basics/data/beCall.py', '\nif __name__ == "__main__":\n    print("beCall")\nelse:\n    print("toCall")')


# - 程式可以知道是**被import還是直接執行**s



get_ipython().system('python "Basics/data/toCall.py"')
# AttributeError: 'NoneType' object has no attribute 'name'
# toCall



get_ipython().system('python "Basics/data/beCall.py"')
# beCall

# ### 試著在beCall中加東西並且讓toCall使用



get_ipython().run_cell_magic('writefile', '_data/beCall.py', 'C=123\ndef test():\n\tprint("321")')




get_ipython().run_cell_magic('writefile', '_data/toCall.py', 'import beCall\nprint(beCall.C)\nbeCall.test()')
# AttributeError: 'NoneType' object has no attribute 'name'
]



get_ipython().system('python _data/toCall.py')
# 123
# 321

### E. SQLite 資料庫
# # SQLite 資料庫(Database)

# - 簡易輕量的資料庫系統
# - 開放原始碼
# - FireFox及Android等軟體也都有內建SQLite
# - SQLite不需要安裝，看起來就只是一個檔案而已

# ### 使用pandas讀取csv資料並存入資料庫中的資料表

# - db的欄位名稱通常不能包含「 `.` 」，所以用「 `_` 」取代



import pandas as pd
iris = pd.read_csv("_data/iris.csv")
iris.columns = iris.columns.str.replace(".","_")


# - 載入sqlite3套件
# - 透過connect連接（工作目錄下的）資料庫"lesson.sqlite"
# - 若原本沒有此資料庫，程式會自動建立一個於工作目錄中



import sqlite3
conn = sqlite3.connect('_data/lesson.sqlite')
# dir(conn)

# - 將iris(DataFrame)存入



iris.to_sql("iris", conn, if_exists="replace")




pd.read_sql_query("SELECT * FROM iris", conn) # [150 rows x 6 columns]



ad=pd.read_csv("_data/Advertising.csv")
ad.iloc[:,1:].to_sql("Advertising",conn, if_exists="replace") # first column is Unnamed: 0




pd.read_sql("SELECT * FROM Advertising",conn) # [200 rows x 5 columns]




pd.read_csv("_data/opendata105Y010.csv").to_sql("opendata",conn,if_exists="replace") # ??




pd.read_sql("SELECT * FROM opendata",conn) # [7852 rows x 13 columns]

### How to get table names using sqlite3 through python? [duplicate]
# https://stackoverflow.com/questions/34570260/how-to-get-table-names-using-sqlite3-through-python?rq=1
res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
type(res) # sqlite3.Cursor
dir(res)

for name in res:
    print (name[0])
#iris
#Advertising
#opendata    

### 參考文獻
# James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013), An Introduction to Statistical Learning with Applications in R, Springer.