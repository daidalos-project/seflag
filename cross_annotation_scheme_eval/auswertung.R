setwd("~/Python_Arbeit/Zwischenlager")

library(dplyr)
library(tidyr)

# Word Form Based Tables
# Maybe needed for different Evaluation
morphol_data_f <- read.csv("cross_corpus_morph_anno_stat_f.csv")
pos_data_f <- read.csv("cross_corpus_pos_stat_f.csv")
lemma_data_f <- read.csv("cross_corpus_lemma_stat_f.csv")

# Lemma Based Tables
morphol_data_l <- read.csv("cross_corpus_morph_anno_stat_l.csv")
pos_data_l <- read.csv("cross_corpus_pos_stat_l.csv")
lemma_data_l <- read.csv("cross_corpus_form_stat_l.csv")

# Auswertung nur POS

# Distribution of POS-Annotations
pos_table <- pos_data_l %>%
  group_by(lemma, annotation, source) %>%
  summarise(count = sum(count), .groups = "drop") %>%
  pivot_wider(
    names_from = source,
    values_from = count,
    values_fill = 0
  )

gesamt_lemma <- pos_table %>%
  group_by(lemma) %>%
  summarise(
    xml_total = sum(xml, na.rm = TRUE),
    conllu_total = sum(conllu, na.rm = TRUE)
  )

kombiniert <- left_join(pos_table, gesamt_lemma, , by = "lemma")
kombiniert2 <- kombiniert %>%
  mutate(
    xml_rel = xml / xml_total,
    conllu_rel = conllu / conllu_total
  )

# Entropie
entropy_per_lemma <- pos_data_l %>%
  group_by(lemma, source) %>%
  summarise(
    total = sum(count),
    p = count / sum(count),     # relative Frequenz
    k = n(),                    # Anzahl unterschiedlicher Annotationen
    entropy_component = -(p * log(p)),
    .groups = "drop"
  ) %>%
  group_by(lemma, source, total) %>%
  summarise(
    entropy = sum(entropy_component) / log(unique(k)),
    .groups = "drop"
  )
entropy_table <- entropy_per_lemma %>%
  mutate(
    entropy = ifelse(is.nan(entropy), 0, entropy) # ersetzt NaN, das durch 0/log(1) entsteht durch 0
  )

View(entropy_table)

entropy_table_wide <- entropy_table %>%
  group_by(lemma, source) %>%
  summarise(entropy = mean(entropy, na.rm = TRUE), .groups = "drop") %>% # mean, damit es nur eine Zeile pro Lemma gibt
  pivot_wider(
    names_from = source,
    values_from = entropy,
    names_prefix = "entropy_"
  )

# Alle Statistiken zusammenführen
final_table <- kombiniert2 %>%
  left_join(entropy_table_wide, by = "lemma")

View(final_table)

entropy_table %>%
  count(lemma) %>%
  filter(n > 1)

# Differenz der Entropie pro Lemma
final_table2 <- final_table %>%
  mutate(entropy_diff = entropy_xml - entropy_conllu)
View(final_table2)

# Auswertung nur mit kompletter morphologischer Annotation

morph_table <- morphol_data_l %>%
  group_by(lemma, annotation, source) %>%
  summarise(count = sum(count), .groups = "drop") # .groups = "drop" hebt die Gruppierung wieder auf

# Gesamte und relative Häufigkeiten pro Lemma
gesamt_lemma_m <- morph_table %>%
  group_by(lemma) %>%
  summarise(
    xml_total = sum(ifelse(source=="xml", count, 0)),
    conllu_total = sum(ifelse(source=="conllu", count, 0))
  )

kombiniert_m <- left_join(morph_table, gesamt_lemma, by="lemma") %>%
  mutate(
    xml_rel = ifelse(source=="xml", count/xml_total, NA),
    conllu_rel = ifelse(source=="conllu", count/conllu_total, NA)
  )

# Entropie pro Lemma
entropy_per_lemma_m <- morph_table %>%
  group_by(lemma, source) %>%
  summarise(
    total = sum(count),
    k = n(),                        # Anzahl verschiedener Annotationen
    entropy = -sum((count/total)*log(count/total))/log(k),
    .groups = "drop"
  ) %>%
  mutate(entropy = ifelse(is.nan(entropy), 0, entropy))

entropy_table_wide_m <- entropy_per_lemma_m %>%
  pivot_wider(names_from = source, values_from = entropy, names_prefix="entropy_")

# Statistiken zusammenführen
final_morph_table <- kombiniert_m %>%
  left_join(entropy_table_wide, by="lemma")

final_morph_table_diff <- final_morph_table %>%
  mutate(entropy_diff = entropy_xml - entropy_conllu)