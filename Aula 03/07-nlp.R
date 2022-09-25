## ---- message = FALSE---------------------------------------------------------
#-----------------------------------------------------------------------
# Pacotes.

library(tm)       # Mineração de texto.
library(udpipe)   # Alguns recursos de NLP.
# ls("package:udpipe")

library(text2vec) # Tokenizers, GloVe, LDA, etc.
# ls("package:text2vec")

# Natural Language Processing.
library(NLP) # Tokenizadores, anotadores, etc.
# ls("package:NLP")

library(openNLP)
ls("package:openNLP")

# TIP: para instalar o pacote treinado para português.
# install.packages("openNLPmodels.pt", repos = "http://datacube.wu.ac.at/")
library(openNLPmodels.pt)

# Informações da sessão.
sessionInfo()


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Segmentação de texto com detecção de sentenças.

# Uma string.
s <- "O Sr. quer mais café? Não. Mas aceito cerveja."
s <- as.String(s)

# Classe e métodos.
class(s)
methods(class = "String")

# Anotador/delimitador de sentenças.
sent_tk_ann <- Maxent_Sent_Token_Annotator(language = "pt")

# Classe e métodos.
class(sent_tk_ann)
methods(class = "Annotator")

# Delimitando a sentença.
a <- sent_tk_ann(s)
a

# Classe, métodos e conteúdo.
class(a)
methods(class = "Annotation")

# Quebrando o documento em sentenças.
cbind(s[a])


## ---- eval = FALSE------------------------------------------------------------
## # Isso irá produzir uma mensagem de erro.
## entity_ann_person <- Maxent_Entity_Annotator(language = "pt",
##                                              kind = "person",
##                                              probs = FALSE,
##                                              model = NULL)


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Rotulação -> determinar as classes gramaticais.

# Anotador/delimitador de palavras.
word_tk_ann <- Maxent_Word_Token_Annotator(language = "pt")
class(word_tk_ann)

# Aplica o anotador sobre o documento.
w <- word_tk_ann(s = s, a = a)
w

# Classe e métodos.
class(w)
methods(class = "Span")

# Quebrando nas sentenças.
cbind(s[w[w$type == "sentence"]])

# Quebrando nas palavras.
cbind(s[w[w$type == "word"]])

# Anotador/rotulador de partes do discurso.
pos_tg_ann <- Maxent_POS_Tag_Annotator(language = "pt")
class(pos_tg_ann)

# Aplica o rotulador (pos tagger).
pos <- pos_tg_ann(s = s, a = w)
pos

# Classe e métodos.
class(pos)
methods(class = "Annotation")

# Cria a tabela com os rótulos.
cbind(token = s[pos], as.data.frame(pos))


## -----------------------------------------------------------------------------
# Sequência de operações para fazer o POS tagging.
annotator_pipeline <-
    Annotator_Pipeline(sent_tk_ann,
                       word_tk_ann,
                       pos_tg_ann)
class(annotator_pipeline)

# Executa as 3 etapas em sequência na string de exemplo.
NLP::annotate(s, annotator_pipeline)

doc <- s
rm(doc)

# Função para fazer em um corpus.
annotateDocuments <- function(doc, pos_filter = NULL) {
    # Documento de classe `String`.
    doc <- as.String(doc)
    # Aplica as anotações.
    doc_with_annotations <- NLP::annotate(doc, annotator_pipeline)
    # Filtra para as palavras.
    ann <- subset(doc_with_annotations, type == "word")
    # Tabela com os tokens.
    tokens <- cbind(token = doc[ann], as.data.frame(ann))
    # Filtra se for especificado.
    if (!is.null(pos_filter)) {
        tokens <- subset(tokens, unlist(features) %in% pos_filter)
    }
    return(tokens)
}

# Termos e suas classes gramaticais.
annotateDocuments(doc = s)

# Só os termos das classes indicadas.
annotateDocuments(doc = s, pos_filter = c("adv", "n", "v-fin"))


## -----------------------------------------------------------------------------
# Faz aquisição do modelo para pt-BR (baixa na hora).
# browseURL("https://github.com/jwijffels/udpipe.models.ud.2.4")
udp_model <- udpipe_download_model(language = 'portuguese')
str(udp_model)

# Carrega o modelo a partir do arquivo `*.uppipe` baixado.
udp_model <- udpipe::udpipe_load_model(file = udp_model$file_model)
class(udp_model)

# Aplicação das anotações em uma string.
x <- udpipe::udpipe_annotate(udp_model, x = as.character(s))
x <- as.data.frame(x)
x[, c("token", "upos", "feats")]


## ---- message = FALSE---------------------------------------------------------
#-----------------------------------------------------------------------
# Pacotes.

library(tidyverse)


## ---- cache = TRUE------------------------------------------------------------
#-----------------------------------------------------------------------
# Importação.

# Dataset com mais de 70 mil registros!
url <- paste0("http://leg.ufpr.br/~walmes/data",
              "/TCC_Brasil_Neto/ImoveisWeb-Realty.csv")
tb <- read_csv2(url, locale = locale(encoding = "latin1"))
str(tb, give.attr = FALSE, vec.len = 1)


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Cria função para determinar a proporção das classes gramaticais por
# documento.

# Aplica no primeiro documento.
s <- head(tb$description, n = 1)
cat(str_wrap(s, width = 72), "\n")

x <- udpipe::udpipe_annotate(udp_model, x = as.character(s))
x <- as.data.frame(x)
str(x)

# Determina o lema e a classe gramatical.
x[, c("token", "lemma", "upos")]

# Converte para tabela de frequência.
as.data.frame(table(x$upos))

# Função para extrair a contagem das classes gramaticais.
getPOStagging_freqs <- function(s) {
    x <- udpipe::udpipe_annotate(udp_model, x = as.character(s))
    pos <- as.data.frame(x)$upos
    as.data.frame(table(pos))
}

#-----------------------------------------------------------------------
# Aplicação em vários documentos.

# Usar amostra para não demorar demais.
tbs <- sample_n(select(tb, description), size = 200)

# Aplicação.
# Essa fase é demorada. Faça com execução paralela.
tbs <- tbs %>%
    mutate(tb_pos = map(description, getPOStagging_freqs)) %>%
    unnest()

# Tabela de frequência.
tbs %>%
    count(pos, wt = Freq, sort = TRUE) %>%
    mutate(f = 100 * n/sum(n))

# Filtrar para as classes relevantes para construção das
# características.
tbs <- tbs %>%
    filter(pos %in% c("NOUN", "VERB", "ADJ")) %>%
    spread(key = "pos", value = "Freq", fill = 0)
tbs[, -1]

biplot(princomp(tbs[, -1], cor = TRUE))


## -----------------------------------------------------------------------------
# Anotador/delimitador de sentenças.
sent_tk_ann <- Maxent_Sent_Token_Annotator(language = "pt")

# Anotador/delimitador de palavras.
word_tk_ann <- Maxent_Word_Token_Annotator(language = "pt")

# Anotador/rotulador de partes do discurso.
pos_tg_ann <- Maxent_POS_Tag_Annotator(language = "pt")

# Sequência de operações para fazer o POS tagging.
annotator_pipeline <-
    Annotator_Pipeline(sent_tk_ann,
                       word_tk_ann,
                       pos_tg_ann)

# Função para fazer em um corpus.
getPOStagging_freqs <- function(doc) {
    # Documento de classe `String`.
    doc <- as.String(doc)
    # Aplica as anotações.
    doc_with_annotations <- NLP::annotate(doc, annotator_pipeline)
    # Filtra para as palavras.
    ann <- subset(doc_with_annotations, type == "word")
    # Tabela com os tokens.
    tokens <- cbind(token = doc[ann], as.data.frame(ann))
    tokens$features <- unlist(tokens$features)
    # Eliminar os detalhamentos: v-pcp, v-fin, v-inf, v-ger, etc.
    tokens$features <- sub("-.*$", "", tokens$features)
    return(as.data.frame(table(tokens$features)))
}

# Usar amostra para não demorar demais.
tbs <- sample_n(select(tb, description), size = 200)

# Aplicando em um documento.
getPOStagging_freqs(tb$description[1])

# Aplicação.
# Essa fase é demorada. Faça com execução paralela.
tbs <- tbs %>%
    mutate(tb_pos = map(description, getPOStagging_freqs)) %>%
    unnest()
names(tbs)

# Tabela de frequência.
tbs %>%
    count(Var1, wt = Freq, sort = TRUE) %>%
    mutate(f = 100 * n/sum(n))

# Filtrar para as classes relevantes para construção das
# características.
tbs <- tbs %>%
    filter(Var1 %in% c("n", "v", "adj")) %>%
    spread(key = "Var1", value = "Freq", fill = 0)
tbs[, -1]

biplot(princomp(tbs[, -1], cor = TRUE))


## ---- message = FALSE, error = FALSE, warning = FALSE-------------------------
#-----------------------------------------------------------------------
# Versões dos pacotes e data do documento.

devtools::session_info()
Sys.time()

