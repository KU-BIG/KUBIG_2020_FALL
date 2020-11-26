pop <- read.csv("C:/Users/924ah/Desktop/경영대_DTB 데이터활용 경진대회/서울시 버스노선별 정류장별 시간대별 승하차 인원 정보.csv", header=T)
head(pop)

loc <- read.csv("C:/Users/924ah/Desktop/경영대_DTB 데이터활용 경진대회/서울특별시 버스정류소 위치정보.csv", header=T)
head(loc)

#####################생략
X <- c()
Y <- c()

for (i in 1:nrow(pop)) {
	for (j in 1:nrow(loc)) {
		if (pop[i,5]==loc[j,1]) {
			X[i] = loc[j,3]
			Y[i] = loc[j,4]
		}else{
			X[i] = 0
			Y[i] = 0
}
}
}

head(X)
head(Y)
length(X)
length(Y)

#inner_join(pop, loc, by=c('))

pop$X <- X
pop$Y <- Y
X <- cbind(X,Y)
head(X)

write.csv(X, file="C:/Users/924ah/Desktop/XY.csv", row.names=FALSE)

####################################

#여기부터
library(tidyverse)

colnames(pop)[1:6] <- c("time", "laneID", "laneName", "ID1", "ID", "Name")
colnames(loc) <- c("ID", "Name", "X", "Y")


## location ID
loc$ID[which(is.na(as.numeric(loc$ID)))] <- c("10016614", "20016614") #원래는 "16614A" "16614B"
loc$ID <- as.numeric(loc$ID)

## passengers ID
pop$ID[which(is.na(as.numeric(pop$ID)))]  # 16614 가 없는 걸로 확인
pop$ID[which(is.na(as.numeric(pop$ID)))] <- "0"
pop$ID <- as.numeric(pop$ID)

location_coord <- loc %>% select(ID, X, Y)
passengers_coord <- left_join(pop, location_coord) 

head(passengers_coord)
lengths(passengers_coord)

pop_sum <- c()
for (i in 1:nrow(passengers_coord)) {
	pop_sum[i] = sum(passengers_coord[i,(7:54)])
}

head(pop_sum)
length(pop_sum)
passengers_coord$pop_sum <- pop_sum

#---------------------------------------------------------------

passengers_coord <- read.csv("C:/Users/924ah/Desktop/passengers_coord.csv", header=T)
passengers_coord <- na.omit(passengers_coord)
head(passengers_coord)

X <- c()
length(X)

passengers_coord[4,5] == passengers_coord[4+1,5]
attach(passengers_coord)
passengers_coord <- passengers_coord[order(ID),]
library(plyr)
passengers_coord <- arrange(passengers_coord, ID)

x <- passengers_coord[1,58]
n <- 1
for (i in 1:nrow(passengers_coord)) {
	if(i+1 > nrow(passengers_coord)) {
		X[n] = x
		break
	}else if(passengers_coord[i,5]==passengers_coord[i+1,5]) {
		x = x + passengers_coord[i+1,58]
	}else if(passengers_coord[i,5]!=passengers_coord[i+1,5]) {
		X[n] = x
		n = n+1
		x = passengers_coord[i+1,58]
}
}


#test용
x <- passengers_coord[1,58]
n <- 1
for (i in 1:84) {
	if(i+1 > 84) {
		print(i)
		X[n] = x
		break
	}else if(passengers_coord[i,5]==passengers_coord[i+1,5]) {
		x = x + passengers_coord[i+1,58]
	}else if(passengers_coord[i,5]!=passengers_coord[i+1,5]) {
		print(i)
		X[n] = x
		n = n+1
		x = passengers_coord[i+1,58]

}
}

sum(passengers_coord[,5]==1006)


# 중복되는 행 제거하고 하나만 남기기
pass_c <- unique(passengers_coord[,c("ID","X","Y")])
head(pass_c)
nrow(pass_c)
length(X)

pass_c <- cbind(pass_c, X)
head(pass_c,10)
colnames(pass_c) <- c("ID", "X", "Y", "f_sum")

write.csv(pass_c, file="C:/Users/924ah/Desktop/버스정류장별_유동인구.csv", row.names=FALSE)