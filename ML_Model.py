import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from joblib import dump, load
df = pd.read_csv('pokemon_data.csv')

COLUMN_NAMES = ['Type 1', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']
TYPES = ['Fire', 'Normal', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Rock', 'Bug', 'Ghost', 'Dark', 'Dragon', 'Steel', 'Fairy']

df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
df = df.loc[df['Total'] > 450]
df = df.loc[~df['Name'].str.contains('Mega')]
df = df.loc[~df['Name'].str.contains('Primal')]

df = df.drop(columns = ['Name'])
df = df.drop(columns = ['Generation'])
df = df.drop(columns = ['Legendary'])
df = df.drop(columns = ['Type 2'])
df = df.drop(columns = ['#'])

df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 0
df.loc[df['Type 1'] == 'Normal', 'Type 1'] = 1
df.loc[df['Type 1'] == 'Water', 'Type 1'] = 2
df.loc[df['Type 1'] == 'Grass', 'Type 1'] = 3
df.loc[df['Type 1'] == 'Electric', 'Type 1'] = 4
df.loc[df['Type 1'] == 'Ice', 'Type 1'] = 5
df.loc[df['Type 1'] == 'Fighting', 'Type 1'] = 6
df.loc[df['Type 1'] == 'Poison', 'Type 1'] = 7
df.loc[df['Type 1'] == 'Ground', 'Type 1'] = 8
df.loc[df['Type 1'] == 'Flying', 'Type 1'] = 9
df.loc[df['Type 1'] == 'Psychic', 'Type 1'] = 10
df.loc[df['Type 1'] == 'Rock', 'Type 1'] = 11
df.loc[df['Type 1'] == 'Bug', 'Type 1'] = 12
df.loc[df['Type 1'] == 'Ghost', 'Type 1'] = 13
df.loc[df['Type 1'] == 'Dark', 'Type 1'] = 14
df.loc[df['Type 1'] == 'Dragon', 'Type 1'] = 15
df.loc[df['Type 1'] == 'Steel', 'Type 1'] = 16
df.loc[df['Type 1'] == 'Fairy', 'Type 1'] = 17

TEMP = ['Type 1']
for col in TEMP:
    df[col] = pd.to_numeric(df[col])
    
df = df.rename(columns = {'Sp. Atk': 'Sp_Atk'})
df = df.rename(columns = {'Sp. Def': 'Sp_Def'})

df_eval_sub = df.loc[df['Total'] < 500]
df_eval_over = df.loc[df['Total'] > 500]      
df = df.drop(columns = ['Total'])
df_eval_sub = df_eval_sub.drop(columns = ['Total'])
y_train = df.pop('Type 1')
y_eval_sub = df_eval_sub.pop('Type 1')
y_eval_over = df_eval_over.pop('Type 1')  

def input_fn(features, labels, training = True, batch_size = 256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)

my_feature_columns = []
for key in df.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units=[30, 10],
    n_classes = 18
)

classifier.train(
    input_fn = lambda: input_fn(df, y_train, training = True),
    steps = 100000
                )

eval_result = classifier.evaluate(input_fn = lambda: input_fn(df_eval_sub, y_eval_sub, training = False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


