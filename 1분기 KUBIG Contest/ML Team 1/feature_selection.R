# 데이터 불러오기
data <- read.csv('C:/Users/924ah/Desktop/Train_data_preprocessed.csv', header=T)
head(data)

#추가 전처리
data
data <- data[,-c(1,3)]
data <- data[,-44]
data <- data[,-93]


r <- as.matrix(cor(data))

col <- c(colnames(data))
col[-1]

length(data)

######lasso
library(glmnet)

x <- as.matrix(data[,-1])
colnames(x) <- col[-1]
head(x)
y <- as.matrix(data$class)

#cv를 통해 lambda 값 정하기
lambdas <- seq(0, 0.5, by=0.001)
cv_fit.lasso <- cv.glmnet(x, y, alpha=1, lambda=lambdas)
plot(cv_fit.lasso) #최적의 값 = 3,4번째 람다 0.003, 0.004

opt_lamb.1 <- .003
opt_lamb.1
opt_lamb.2 <- .004
opt_lamb.2

fin.lasso1 <- glmnet(x, y, alpha=1, lambda=opt_lamb.1)
coef(fin.lasso1) #31개
list_1 <- c('aj_000', 'al_000', 'am_0', 'aq_000', 'ar_000','at_000','au_000','av_000','bi_000','bj_000','bs_000','by_000','cb_000','cg_000','ci_000','cj_000','cm_000','cz_000','de_000','dg_000','di_000','do_000','du_000','dx_000','dy_000','ay_Var','ba_mean','cn_mean','cn_Var','cs_sum','ee_mean')


fin.lasso2 <- glmnet(x, y, alpha=1, lambda=opt_lamb.2)
coef(fin.lasso2) #24개
list_2 <- c('al_000', 'am_0', 'aq_000', 'ar_000','at_000','bj_000','bs_000','by_000',
		'cg_000','ci_000','cj_000','cm_000','de_000','dg_000','di_000','do_000','dx_000','dy_000','ay_Var',
		'ba_mean', 'cn_mean','cn_Var','cs_sum','ee_mean')

data.1 <- data[,list_1]
data.2 <- data[,list_2]

cor.1 <- cor(data.1)
cor.2 <- cor(data.2)

cor.1 >= 0.7 
#-> am_0&al_000 / bi_000&aq_000 / bj_000&bi_000 / bj_000&aq_000 / by_000&aq_000 / cb_000&bs_000 / ci_000&aq,bi,bj,by
cor.2