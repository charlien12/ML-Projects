import pandas as pd

# Auto-detect delimiter
student_math_df = pd.read_csv('data/Maths.csv', sep=',', encoding='ISO-8859-1')

print(student_math_df.head())
# Display basic information about the DataFrame
print(student_math_df.isnull().sum())
print(student_math_df.duplicated().sum())
print(student_math_df['G3'].value_counts())