(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
   {:format              [:quarto :html]
    :book                {:title "M7550: Final Project"}
    :subdirs-to-sync     ["notebooks" "data" "images"]
    :source-path         ["src/index.clj"
                          "notebooks/python/features.ipynb"
                          "notebooks/python/model.ipynb"
                          "notebooks/technical_report.md"]
    :base-target-path    "docs"
    :clean-up-target-dir true}))

(comment
  (build))
