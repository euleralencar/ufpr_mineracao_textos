## ---- message = FALSE---------------------------------------------------------
library(tm)
library(slam)
library(text2vec)
library(proxy)
library(RWeka)
library(Matrix)
library(tidyverse)


## -----------------------------------------------------------------------------


# cname <- file.path("C:\\Users\\euler\\OneDrive\\Documents\\Projetos\\projetos_r\\2022\\ufpr_mineracao_textos\\Aula 02", "respostas-prova")

data <- read.csv(file = "Exercicio_agrupamento/eleicao.csv")

data <- head(data, 3)

# data %>% View()

cname <- data %>% select(titulo, ementa)

head(cname)

# cname <- data %>% select(ementa) 
# cname <- data %>% select(ementa) %>% as.vector()

# cps <- VCorpus(DirSource(cname, 
#                           encoding = "UTF-8"), 
#                 readerControl=list(reader=readPlain))

df_source <- DataframeSource(cname)

cps <- VCorpus(VectorSource(cname))

names(cps) <- sub(".*(.{3})\\.txt", "\\1", names(cps))
inspect(cps[[3]])


# # Não funcionou
# cps <- VCorpus(tm::ZipSource("Aula 02/respostas-prova.zip", 
#                              mode = "text",
#                              pattern = "*.txt"),
#                readerControl = list(language = "portuguese"))
# 
# names(cps) <- sub(".*(.{3})\\.txt", "\\1", names(cps))
# cps
# 
# inspect(cps[[3]])


# bar <- iconv((cps), to="UTF-8")
# 
# inspect(bar[[3]])
# 
# iconv(cps, from = "ISO_8859-2", to = "")
# 
# inspect(cps[[3]])


## -----------------------------------------------------------------------------
# Fazendo as operações de limpeza.
cps <- tm_map(cps, FUN = content_transformer(tolower))
cps <- tm_map(cps, FUN = removePunctuation)
cps <- tm_map(cps, FUN = removeNumbers)
cps <- tm_map(cps, FUN = removeWords, words = stopwords("portuguese"))
cps <- tm_map(cps, FUN = stripWhitespace)
cps <- tm_map(cps, FUN = stemDocument, language = "portuguese")
cps <- tm_map(cps, FUN = content_transformer(trimws))




# Passa para ASCII puro.
# acen <- function(x) iconv(x, to = "ASCII//TRANSLIT")
# cps <- tm_map(cps, FUN = content_transformer(acen))
# sapply(cps[1:4], content)

# inspect(cps[[3]])

## -----------------------------------------------------------------------------
# Um tokenizador de bi-gramas.
my_tokenizer <- function(x) {
    RWeka::NGramTokenizer(x, control = Weka_control(min = 2, max = 2))
}

tt <- Token_Tokenizer(my_tokenizer)
tt("Minha terra tem palmeiras onde canta o sabiá.")

dtm <- DocumentTermMatrix(cps, control = list(tokenize = tt))
dtm

sample(Terms(dtm), size = 10)
# content(cps[[1]])

# Matriz menos esparsa.
rst <- removeSparseTerms(dtm, sparse = 0.75)
rst


## -----------------------------------------------------------------------------
# Transforma em matriz ordinária.
m <- as.matrix(rst)

# Distância coseno entre documentos.
d_mat <- text2vec::dist2(m, method = "cosine")
str(d_mat)

# De matriz cheia para triangular inferior.
d_mat <- stats::as.dist(d_mat)
str(d_mat)


## -----------------------------------------------------------------------------
# Faz o dendrograma.
hc <- hclust(d_mat, method = "average")
plot(hc, hang = -1)

# ATTENTION: cuidado em fazer isso com dimensões proibitivas.
u <- which(as.matrix(d_mat) == min(d_mat), arr.ind = TRUE)
i <- names(cps) %in% rownames(u)
names(cps)[which(i)]

purrr::walk(
    as.list(cps[which(i)]),
    function(x) {
      u <- content(x)
      cat("-------------------------------",
          strwrap(paste(u, collapse = " "),
                  width = 60),
          sep = "\n")
  })


## ---- message=FALSE, error=FALSE, warning=FALSE-------------------------------
#-----------------------------------------------------------------------
# Versões dos pacotes e data do documento.

devtools::session_info()
Sys.time()

