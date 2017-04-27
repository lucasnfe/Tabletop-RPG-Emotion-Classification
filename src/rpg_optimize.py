#-----------------------------------------------------------
# Name: rpg_density.py
# Calculate Emotion Word Density using NRC Emotion Lexicon
# Author: Lucas N. Ferreira
# E-mail: lferreira@ucsc.edu
#-----------------------------------------------------------

import sys
import rpg_ec
import rpg_density

def optimize_emotion_mapping(lexicon, narrative):
    counter = 0
    max_accuracy = [{}, 0]
    for a in rpg_ec.RPG_EMOTIONS:
        for b in rpg_ec.RPG_EMOTIONS:
            for c in rpg_ec.RPG_EMOTIONS:
                for d in rpg_ec.RPG_EMOTIONS:
                    for e in rpg_ec.RPG_EMOTIONS:
                        for f in rpg_ec.RPG_EMOTIONS:
                            for g in rpg_ec.RPG_EMOTIONS:
                                for h in rpg_ec.RPG_EMOTIONS:
                                    emotion_map = {
                                        "anger"       : a,
                                        "fear"        : b,
                                        "sadness"     : c,
                                        "disgust"     : d,
                                        "joy"         : e,
                                        "trust"       : f,
                                        "surprise"    : g,
                                        "anticipation": h,
                                    }
                                    counter += 1

                                    classification = rpg_density.classify_narrative(narrative, lexicon, emotion_map, 20)

                                    # Calculate confusion matrix
                                    confusion_matrix = rpg_ec.rpg_create_confusion_matrix(narrative, classification)

                                    print(emotion_map)
                                    rpg_ec.rpg_print_confusion_matrix(confusion_matrix)
                                    accuracy = rpg_ec.calculate_accuracy(confusion_matrix)
                                    if accuracy > max_accuracy[1]:
                                        max_accuracy[0] = emotion_map
                                        max_accuracy[1] = accuracy

                                    print(accuracy)
                                    print("-------------")

    print(counter)
    print(max_accuracy)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(sys.argv[0] + " <lexicon_filepath> " + "<narrative_filepath>")
        quit()

    lexicon_filepath = sys.argv[1]
    narrative_filepath = sys.argv[2]

    # Lexicon is a dict indexed by words of the english dictionary.
    # Each value is a list of tuples [(nrc_emotion, association)]
    lexicon = rpg_density.parse_nrc_lexicon(lexicon_filepath)

    # Narrative is a dict indexed by sentence id.
    # Each value is a tuple (sentence, rpg_emotion)
    narrative = rpg_ec.parse_narrative_data(narrative_filepath)

    optimize_emotion_mapping(lexicon, narrative)
