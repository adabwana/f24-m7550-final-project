(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(def r-host "r")  ; container name becomes hostname
(def python-host "python")


(defn build []
  (clay/make!
   {:format              [:quarto :html]
    :book                {:title "M7550: Final Project"}
    :subdirs-to-sync     ["notebooks" "data" "images"]
    :source-path         ["src/index.clj"
                          ;; "notebooks/python/features.ipynb"
                          ;; "notebooks/python/model.ipynb"
                          "notebooks/r/Final project.Rmd"]
    :base-target-path    "docs"
    :clean-up-target-dir true
    :engines             {:r      {:host r-host
                                   :port 6311}
                          :python  {:host python-host
                                    :port 8888}}}))

(comment
  (build))
