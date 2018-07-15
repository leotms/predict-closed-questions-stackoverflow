'''
    Author: Leonardo Martinez.
    Last Updated: 13/07/2018.
    Plots feature importance for both RF and LightGBM best models.
'''
import matplotlib.pyplot as plt

# Best features Random Forest
attributes = ['BodyLenght', 'FirstSentenceLenght', 'InterrogativeWordsInBody',
              'InterrogativeWordsInTitle', 'IsTitleAQuestion', 'LowerLettersRatioInBody',
              'NumberBodyDigits', 'NumberBodyLinks', 'NumberBodySentences', 'NumberBodySentencesWithI',
              'NumberBodySentencesWithYou', 'NumberOfTags', 'OwnerAge', 'PostAge', 'TitleLenght',
              'UpperLettersRatioInBody', 'UserDownVotes', 'UserUpVotes', 'UserViews']

best_rf = [8.69777806e-03 ,3.56182389e-03 ,1.22531310e-02, 7.71368178e-04,
           1.02605455e-03 ,1.10310566e-02, 3.70151185e-01 ,9.14465945e-04,
           5.91268705e-03 ,5.01838562e-03, 4.42525233e-05 ,1.52584620e-03,
           1.24083851e-02, 5.71604574e-03, 3.52030947e-03 ,4.04507402e-03,
           2.16484386e-01, 2.00461415e-01 ,1.36456350e-01]

plt.figure()
plt.bar([i for i in range(19)], best_rf, color='#16a085')
plt.title('Importancia del los Atributos para RandomForest')
plt.xticks([i for i in range(19)], attributes, rotation='vertical')
plt.tight_layout()
plt.savefig('graphs/rf_attributes.png')
plt.show()
plt.close()

best_lgbm = [9.87370773e+03 ,6.02106099e+03 ,8.02875137e+03 ,6.27785905e+02,
             1.64240492e+03, 1.65341700e+04 ,1.49414118e+05 ,1.24241389e+03,
             4.03897558e+03, 7.26818215e+03 ,1.29769953e+02, 2.03935412e+03,
             7.59948443e+03 ,9.70253538e+03, 6.99043868e+03 ,7.31704176e+03,
             6.60103895e+04, 1.85289167e+05 ,1.80992510e+04]

plt.figure()
plt.bar([i for i in range(19)], best_lgbm, color='#16a085')
plt.title('Importancia del los Atributos para LightGBM')
plt.xticks([i for i in range(19)], attributes, rotation='vertical')
plt.tight_layout()
plt.savefig('graphs/lightgbm_attributes.png')
plt.show()
