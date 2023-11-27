{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "e3bfafa0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "7adaadf1",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv(r\"C:\\Users\\ameya\\Downloads\\Spam_Data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "9bdbe3d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category                                            Message\n",
       "0      ham  Go until jurong point, crazy.. Available only ...\n",
       "1      ham                      Ok lar... Joking wif u oni...\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      ham  U dun say so early hor... U c already then say...\n",
       "4      ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "e6aa80a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5572, 2)"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "52740b8e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5572 entries, 0 to 5571\n",
      "Data columns (total 2 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   Category  5572 non-null   object\n",
      " 1   Message   5572 non-null   object\n",
      "dtypes: object(2)\n",
      "memory usage: 87.2+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "59e8d012",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Category    0\n",
       "Message     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "6cd92439",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "403"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "6790434d",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=data.drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "bb596211",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5169, 2)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "af9fbd5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3237</th>\n",
       "      <td>ham</td>\n",
       "      <td>Good. No swimsuit allowed :)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4568</th>\n",
       "      <td>ham</td>\n",
       "      <td>At WHAT TIME should i come tomorrow</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4164</th>\n",
       "      <td>spam</td>\n",
       "      <td>Dear Voucher Holder, To claim this weeks offer...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2947</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nope but i'll b going 2 sch on fri quite early...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1096</th>\n",
       "      <td>spam</td>\n",
       "      <td>Dear Subscriber ur draw 4 å£100 gift voucher w...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1307</th>\n",
       "      <td>spam</td>\n",
       "      <td>Get 3 Lions England tone, reply lionm 4 mono o...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Category                                            Message\n",
       "3237      ham                       Good. No swimsuit allowed :)\n",
       "4568      ham                At WHAT TIME should i come tomorrow\n",
       "4164     spam  Dear Voucher Holder, To claim this weeks offer...\n",
       "2947      ham  Nope but i'll b going 2 sch on fri quite early...\n",
       "1096     spam  Dear Subscriber ur draw 4 å£100 gift voucher w...\n",
       "1307     spam  Get 3 Lions England tone, reply lionm 4 mono o..."
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.sample(6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "d9e5d46b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['ham', 'spam'], dtype=object)"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Category'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "bca9a525",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Category', ylabel='count'>"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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nHPq1KkGr3v69OkGV5++KTAw0Pn028UHAAC4Pvk8iC5ljFFtba169Oih6Oho7dq1y9lXV1ennJwcDR48WJLUv39/tW/f3mtOcXGxDh8+7MxJSkqSx+PRgQMHnDl5eXnyeDzOHAAAYDd/X774v/zLv+iuu+5SbGysKisrlZmZqb179yo7O1sul0upqalasmSJ4uLiFBcXpyVLlig4OFjJycmSJLfbrWnTpmnevHmKiIhQeHi45s+fr8TERI0YMUKS1KdPH40ZM0bTp0/X+vXrJUkPPfSQxo0bd9WfMAMAANc3nwbRqVOnlJKSouLiYrndbvXr10/Z2dkaOXKkJCktLU01NTWaOXOmysrKNHDgQO3cuVOhoaHOMVauXCl/f39NmjRJNTU1Gj58uDZt2iQ/Pz9nztatWzVnzhzn02gTJkzQ6tWrW/ZkAQBAq9XqvoeoteJ7iADf4XuIADRWm/seIgAAAF8hiAAAgPUIIgAAYD2CCAAAWI8gAgAA1iOIAACA9QgiAABgPYIIAABYjyACAADWI4gAAID1CCIAAGA9gggAAFiPIAIAANYjiAAAgPUIIgAAYD2CCAAAWI8gAgAA1iOIAACA9QgiAABgPYIIAABYjyACAADWI4gAAID1CCIAAGA9gggAAFiPIAIAANZrVBDdeeedKi8vbzBeUVGhO++881rXBAAA0KIaFUR79+5VXV1dg/GvvvpK77777jUvCgAAoCX5f5/JH374ofO///KXv6ikpMTZrq+vV3Z2trp169Z0qwMAAGgB3yuIfvzjH8vlcsnlcl32rbGgoCC98MILTbY4AACAlvC9gqioqEjGGPXs2VMHDhxQly5dnH0BAQGKjIyUn59fky8SAACgOX2vILrxxhslSRcuXGiWxQAAAPjC9wqib/rkk0+0d+9elZaWNgikp5566poXBgAA0FIaFUQbN27UI488os6dOys6Oloul8vZ53K5CCIAANCmNCqInn32Wf3mN7/RggULmno9AAAALa5R30NUVlamiRMnNvVaAAAAfKJRQTRx4kTt3LmzqdcCAADgE416y6xXr1568sknlZubq8TERLVv395r/5w5c5pkcQAAAC2hUUG0YcMGdejQQTk5OcrJyfHa53K5CCIAANCmNCqIioqKmnodAAAAPtOoe4gAAACuJ426QjR16tQr7n/ppZcatRgAAABfaFQQ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Ki0tlSTt2bNHI0eOVLdu3RQaGqr77rtPZ86cUXV1tTM/ODhYN998s7MdFRWlm266SR06dPAau3hMAE3vlltu0fDhw5WYmKiJEydq48aNKisr89ofHBzsbCclJamqqkrHjx+XJB07dkzJycnq2bOnwsLC1KNHD0nSl19+6fU63/zvRVRUlCQpMTHRa4zf9esPQQQrtG/f3mvb5XLpwoUL+uKLLzR27FglJCTozTffVH5+vtasWSNJOnfu3BWf/23HBNA8/Pz8tGvXLu3YsUN9+/bVCy+8oPj4eBUVFV3xeS6XS5I0fvx4nTlzRhs3blReXp7y8vIkyes+I8n79/3icy8d43f9+mPNH3cFLufQoUM6f/68VqxYoXbtvv7/B6+//rqPVwXg27hcLg0ZMkRDhgzRU089pRtvvFFZWVmSpP/5n/9RTU2NgoKCJEm5ubnq0KGDunfvrjNnzqiwsFDr16/X7bffLknat2+fz84DrQ9BBKvdfPPNOn/+vF544QWNHz9e7733ntatW+frZQG4jLy8PO3evVujRo1SZGSk8vLydPr0afXp00cffvih6urqNG3aND3xxBP64osvtGjRIj366KNq166dOnXqpIiICG3YsEFdu3bVl19+qV/96le+PiW0IrxlBqv9+Mc/VkZGhpYuXaqEhARt3bpV6enpvl4WgMsICwvTO++8o7Fjx6p379564okntGLFCt11112SpOHDhysuLk533HGHJk2apPHjxztfsdGuXTtlZmYqPz9fCQkJeuyxx7R8+XIfng1aG5cx//95RQAA2qgHHnhA5eXl+uMf/+jrpaCN4goRAACwHkEEAACsx1tmAADAelwhAgAA1iOIAACA9QgiAABgPYIIAABYjyACAADWI4gAAID1CCIArU5JSYlmz56tnj17KjAwULGxsRo/frx27959Vc/ftGmTOnbs2LyLBHBd4Y+7AmhVPv/8cw0ZMkQdO3bUsmXL1K9fP507d07/9V//pVmzZunIkSO+XuL3du7cObVv397XywBwBVwhAtCqzJw5Uy6XSwcOHNAvfvEL9e7dWz/60Y80d+5c5ebmSpIyMjKUmJiokJAQxcbGaubMmaqqqpIk7d27V//0T/8kj8cjl8sll8vl/IHPuro6paWlqVu3bgoJCdHAgQO1d+9er9ffuHGjYmNjFRwcrHvvvVcZGRkNrjatXbtWN998swICAhQfH68tW7Z47Xe5XFq3bp1+9rOfKSQkRM8++6x69eql3/72t17zDh8+rHbt2unYsWNN9wME0DgGAFqJM2fOGJfLZZYsWXLFeStXrjRvv/22+etf/2p2795t4uPjzSOPPGKMMaa2ttY899xzJiwszBQXF5vi4mJTWVlpjDEmOTnZDB482Lzzzjvms88+M8uXLzeBgYHmk08+McYYs2/fPtOuXTuzfPlyc/ToUbNmzRoTHh5u3G6389rbtm0z7du3N2vWrDFHjx41K1asMH5+fubtt9925kgykZGR5ve//705duyY+fzzz81vfvMb07dvX6/zeOyxx8wdd9zRFD86ANeIIALQauTl5RlJZtu2bd/rea+//rqJiIhwtl9++WWviDHGmM8++8y4XC7zv//7v17jw4cPNwsXLjTGGDN58mRz9913e+2fMmWK17EGDx5spk+f7jVn4sSJZuzYsc62JJOamuo15+TJk8bPz8/k5eUZY4ypq6szXbp0MZs2bfpe5wqgefCWGYBWw/z/n1Z0uVxXnLdnzx6NHDlS3bp1U2hoqO677z6dOXNG1dXV3/qcDz74QMYY9e7dWx06dHAeOTk5zltWR48e1U9/+lOv5126XVhYqCFDhniNDRkyRIWFhV5jAwYM8Nru2rWr7r77br300kuSpLfeektfffWVJk6ceMVzBdAyCCIArUZcXJxcLleDuPimL774QmPHjlVCQoLefPNN5efna82aNZK+vnn521y4cEF+fn7Kz89XQUGB8ygsLNTzzz8v6esguzTGzGX+/vXl5lw6FhIS0uB5Dz74oDIzM1VTU6OXX35ZkydPVnBw8LeuGUDLIYgAtBrh4eEaPXq01qxZc9mrPeXl5Tp06JDOnz+vFStWaNCgQerdu7dOnjzpNS8gIED19fVeYz/5yU9UX1+v0tJS9erVy+sRHR0tSfrhD3+oAwcOeD3v0KFDXtt9+vTRvn37vMb279+vPn36fOf5jR07ViEhIVq7dq127NihqVOnfudzALQMgghAq/Liiy+qvr5eP/3pT/Xmm2/q008/VWFhoVatWqWkpCTdfPPNOn/+vF544QX99a9/1ZYtW7Ru3TqvY9x0002qqqrS7t279be//U1nz55V7969NWXKFN13333atm2bioqKdPDgQS1dulTbt2+XJM2ePVvbt29XRkaGPv30U61fv147duzwuvrz+OOPa9OmTVq3bp0+/fRTZWRkaNu2bZo/f/53npufn58eeOABLVy4UL169VJSUlLT/vAANJ5P72ACgMs4efKkmTVrlrnxxhtNQECA6datm5kwYYLZs2ePMcaYjIwM07VrVxMUFGRGjx5tXnnlFSPJlJWVOcd4+OGHTUREhJFkFi1aZIz5+kbmp556ytx0002mffv2Jjo62tx7773mww8/dJ63YcMG061bNxMUFGTuuece8+yzz5ro6Giv9b344oumZ8+epn379qZ3797mlVde8dovyWRlZV323I4dO2YkmWXLll3zzwlA03EZc5k3yAEAkqTp06fryJEjevfdd5vkeO+9956GDRumEydOKCoqqkmOCeDa8U3VAPANv/3tbzVy5EiFhIRox44d2rx5s1588cVrPm5tba2OHz+uJ598UpMmTSKGgFaGe4gA4BsOHDigkSNHKjExUevWrdOqVav04IMPXvNxX3vtNcXHx8vj8WjZsmVNsFIATYm3zAAAgPW4QgQAAKxHEAEAAOsRRAAAwHoEEQAAsB5BBAAArEcQAQAA6xFEAADAegQRAACw3v8BHTiFKTlDEXIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "c=data['Category'].value_counts()\n",
    "sns.countplot(x='Category',data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "1b484507",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "([<matplotlib.patches.Wedge at 0x20ad8137dd0>,\n",
       "  <matplotlib.patches.Wedge at 0x20adb84da90>],\n",
       " [Text(-1.0144997251399075, 0.4251944351600247, 'ham'),\n",
       "  Text(1.014499764949479, -0.4251943401757036, 'spam')],\n",
       " [Text(-0.5533634864399495, 0.23192423736001344, '87.37%'),\n",
       "  Text(0.5533635081542612, -0.23192418555038377, '12.63%')])"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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esUGo1dcEHrFBqFLZCB0vGiNZIoFQYpRVmf5Q2lklTfIqjiMei1ign0QF+KHHjGBLWUQG4SesUFcU4EUhwVBimexOVDdaIWpwYrqRiuqzC3/rjJbYTJbUW0++e9mVJmtMDc7AA2gAaDRAFqN5td/a6A59XkNtJpfPwcgwEePMH8DwgKMCPc3ItTfiPAAAyICfRAb7IuYYF/EhvgiPMAo9ptB5EYsCCIiconj5YiIyCUWBBERucSCICIil1gQRETkEguCiIhcYkEQEZFLLAgiInKJBUFERC6xIIiIyCUWBBERucSCICIil1gQRETkEguCiIhcYkEQEZFLLAgiInKJBUFERC6xIIiIyCUWBBERucSCICIil1gQRETkEguCiIhcYkEQEZFLLAgiInKJBUFERC6xIIiIyCUWBBERucSCICIil1gQRETkEguCiIhcYkEQEZFLLAgiInKJBUFERC6xIIiIyCUWBBERucSCICIil1gQRETkEguCiIhcYkEQEZFLLAgiInKJBUFERC6xIIiIyCUWBBERucSCICIil1gQRETkEguCiIhcYkEQEZFL/w9e06GAn7BzzgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.pie(c,labels=['ham','spam'],autopct='%1.2f%%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "57ae55c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.loc[data['Category']=='spam','Category']=0\n",
    "data.loc[data['Category']=='ham','Category']=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "e4d05f36",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>0</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>1</td>\n",
       "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>1</td>\n",
       "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>1</td>\n",
       "      <td>The guy did some bitching but I acted like i'd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>1</td>\n",
       "      <td>Rofl. Its true to its name</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5169 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Category                                            Message\n",
       "0           1  Go until jurong point, crazy.. Available only ...\n",
       "1           1                      Ok lar... Joking wif u oni...\n",
       "2           0  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3           1  U dun say so early hor... U c already then say...\n",
       "4           1  Nah I don't think he goes to usf, he lives aro...\n",
       "...       ...                                                ...\n",
       "5567        0  This is the 2nd time we have tried 2 contact u...\n",
       "5568        1              Will Ì_ b going to esplanade fr home?\n",
       "5569        1  Pity, * was in mood for that. So...any other s...\n",
       "5570        1  The guy did some bitching but I acted like i'd...\n",
       "5571        1                         Rofl. Its true to its name\n",
       "\n",
       "[5169 rows x 2 columns]"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "144d34c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0], dtype=object)"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Category'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "4ce03e59",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "802e6dcd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\ameya\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "e95683eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['num_characters']=data['Message'].apply(len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "ef010c6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "      <th>num_characters</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category                                            Message  num_characters\n",
       "0        1  Go until jurong point, crazy.. Available only ...             111\n",
       "1        1                      Ok lar... Joking wif u oni...              29\n",
       "2        0  Free entry in 2 a wkly comp to win FA Cup fina...             155\n",
       "3        1  U dun say so early hor... U c already then say...              49\n",
       "4        1  Nah I don't think he goes to usf, he lives aro...              61"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0660a4ec",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "cbe47e99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "      <th>num_characters</th>\n",
       "      <th>num_words</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "      <td>24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "      <td>37</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category                                            Message  num_characters  \\\n",
       "0        1  Go until jurong point, crazy.. Available only ...             111   \n",
       "1        1                      Ok lar... Joking wif u oni...              29   \n",
       "2        0  Free entry in 2 a wkly comp to win FA Cup fina...             155   \n",
       "3        1  U dun say so early hor... U c already then say...              49   \n",
       "4        1  Nah I don't think he goes to usf, he lives aro...              61   \n",
       "\n",
       "   num_words  \n",
       "0         24  \n",
       "1          8  \n",
       "2         37  \n",
       "3         13  \n",
       "4         15  "
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['num_words']=data['Message'].apply(lambda x:len(nltk.word_tokenize(x)))\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "799f189d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "232e553e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "      <th>num_characters</th>\n",
       "      <th>num_words</th>\n",
       "      <th>num_sentences</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "      <td>24</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "      <td>37</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "      <td>13</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "      <td>15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category                                            Message  num_characters  \\\n",
       "0        1  Go until jurong point, crazy.. Available only ...             111   \n",
       "1        1                      Ok lar... Joking wif u oni...              29   \n",
       "2        0  Free entry in 2 a wkly comp to win FA Cup fina...             155   \n",
       "3        1  U dun say so early hor... U c already then say...              49   \n",
       "4        1  Nah I don't think he goes to usf, he lives aro...              61   \n",
       "\n",
       "   num_words  num_sentences  \n",
       "0         24              2  \n",
       "1          8              2  \n",
       "2         37              2  \n",
       "3         13              1  \n",
       "4         15              1  "
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['num_sentences']=data['Message'].apply(lambda x:len(nltk.sent_tokenize(x)))\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f4db442",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "e432eff7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>num_characters</th>\n",
       "      <th>num_words</th>\n",
       "      <th>num_sentences</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5169.000000</td>\n",
       "      <td>5169.000000</td>\n",
       "      <td>5169.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>78.977945</td>\n",
       "      <td>18.455794</td>\n",
       "      <td>1.965564</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>58.236293</td>\n",
       "      <td>13.324758</td>\n",
       "      <td>1.448541</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>36.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>60.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>117.000000</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>910.000000</td>\n",
       "      <td>220.000000</td>\n",
       "      <td>38.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       num_characters    num_words  num_sentences\n",
       "count     5169.000000  5169.000000    5169.000000\n",
       "mean        78.977945    18.455794       1.965564\n",
       "std         58.236293    13.324758       1.448541\n",
       "min          2.000000     1.000000       1.000000\n",
       "25%         36.000000     9.000000       1.000000\n",
       "50%         60.000000    15.000000       1.000000\n",
       "75%        117.000000    26.000000       2.000000\n",
       "max        910.000000   220.000000      38.000000"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[['num_characters','num_words','num_sentences']].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "620c3115",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>num_characters</th>\n",
       "      <th>num_words</th>\n",
       "      <th>num_sentences</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>653.000000</td>\n",
       "      <td>653.000000</td>\n",
       "      <td>653.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>137.891271</td>\n",
       "      <td>27.667688</td>\n",
       "      <td>2.970904</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>30.137753</td>\n",
       "      <td>7.008418</td>\n",
       "      <td>1.488425</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>13.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>132.000000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>149.000000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>157.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>224.000000</td>\n",
       "      <td>46.000000</td>\n",
       "      <td>9.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       num_characters   num_words  num_sentences\n",
       "count      653.000000  653.000000     653.000000\n",
       "mean       137.891271   27.667688       2.970904\n",
       "std         30.137753    7.008418       1.488425\n",
       "min         13.000000    2.000000       1.000000\n",
       "25%        132.000000   25.000000       2.000000\n",
       "50%        149.000000   29.000000       3.000000\n",
       "75%        157.000000   32.000000       4.000000\n",
       "max        224.000000   46.000000       9.000000"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['Category']==0][['num_characters','num_words','num_sentences']].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "10d19108",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>num_characters</th>\n",
       "      <th>num_words</th>\n",
       "      <th>num_sentences</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>4516.000000</td>\n",
       "      <td>4516.000000</td>\n",
       "      <td>4516.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>70.459256</td>\n",
       "      <td>17.123782</td>\n",
       "      <td>1.820195</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>56.358207</td>\n",
       "      <td>13.493970</td>\n",
       "      <td>1.383657</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>34.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>52.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>90.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>910.000000</td>\n",
       "      <td>220.000000</td>\n",
       "      <td>38.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       num_characters    num_words  num_sentences\n",
       "count     4516.000000  4516.000000    4516.000000\n",
       "mean        70.459256    17.123782       1.820195\n",
       "std         56.358207    13.493970       1.383657\n",
       "min          2.000000     1.000000       1.000000\n",
       "25%         34.000000     8.000000       1.000000\n",
       "50%         52.000000    13.000000       1.000000\n",
       "75%         90.000000    22.000000       2.000000\n",
       "max        910.000000   220.000000      38.000000"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['Category']==1][['num_characters','num_words','num_sentences']].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3938c1f3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "832dab0e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='num_characters', ylabel='Count'>"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Zp8HDhyQy+XSmDFjWv3z+vTpI0nm/t1ut5599lkNGzZMMTEx6tGjhzZv3qyjR4967WPEiBHN9rt9+3aNGzdOffv2VWRkpB5++GGdOHFCp0+flvTvkZ22yM7O1vbt29WjRw9zufrqqyVJX3zxRYv1fPe731Vtba0SEhL06KOP6u233zZHrwBcHGEHgCkoKMjrvc1mk8fjUUDA1/9UGIZhbquvr7/oPmw2W4v7vJiwsLA212yz2STJ3P8LL7ygZcuWaf78+dq2bZtycnI0fvz4ZpN+IyIivN4fOXJEd955p1JSUvTHP/5R2dnZWrVqlaR/H3dr62vM4/Ho7rvvVk5Ojtdy8OBB3XrrrS3WEx8fr/z8fK1atUphYWGaPn26br311hb7AIA3wg6Ai7ryyislSSUlJea6zr6/TWJiosLCwrR169Z27+P999/Xd77zHT344IO69tprlZCQoIMHD170c5988okaGhr0wgsv6Nvf/rauuuoqffnll15thg0bdsHagoOD5Xa7vdZ985vfVF5engYOHKjBgwd7LU0DTlNhYWGaOHGiXnrpJe3YsUMffPCB9u3bd9FjAUDYAdAKYWFh+va3v63nnntO+/fv19///nf97Gc/69SfGRoaqieeeELz58/Xr371K33xxRf68MMPtWbNmlbvY/DgwcrKytLu3bt14MABTZs2TaWlpRf93De+8Q01NDRoxYoVOnTokN5880298sorXm0WLFigPXv2aPr06dq7d68+//xzrV692ryyauDAgfroo490+PBhlZeXy+PxaMaMGTp58qSmTJmijz/+WIcOHdLmzZv1/e9/v1kwamzt2rVas2aNcnNzzXrCwsI0YMCAVv8ugMsZYQdAq/zyl79UfX29RowYodmzZ2vRokWd/jOffPJJzZ07Vz//+c91zTXX6N57723THKInn3xS3/zmNzV+/HiNHj1aDodD99xzz0U/d91112np0qV6/vnnlZKSol//+tfKyMjwanPVVVdp8+bN+uyzz/Stb31LN910k/785z/LbrdLkubNm6fAwEANGTJEV155pY4ePSqn06l//OMfcrvdGj9+vFJSUjR79mxFR0ebpwrP54orrtDrr7+um2++2RxReueddxQTE9Pq3wVwObMZjU/CAwAAWAwjOwAAwNIIOwB84ujRo16XYDddml4eDgDtxWksAD7R0NBgPkLhfAYOHGjOfwGAS0HYAQAAlsZpLAAAYGmEHQAAYGmEHQAAYGmEHQAAYGmEHQAAYGmEHQAAYGmEHQAAYGmEHQAAYGn/H31uJgolQoOhAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data[data['Category']==1]['num_characters'])\n",
    "sns.histplot(data[data['Category']==0]['num_characters'],color='r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "e9f58804",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='num_words', ylabel='Count'>"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAGxCAYAAACEFXd4AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA1LklEQVR4nO3dfVhU953//9fInYAw3qCMKCoxGIOYxGjWaNqKUUltjUm9rmpi2zW7Nj+tNylVa+PaNtRNMLXxZiu52WRtNHEtbb+NXXebGjEqjWXTRaKJGE1oJFEUQjWEAcQZmDm/PyynDDcKODDD8fm4rnOFc87nnHmfOeC8cs7nzMdmGIYhAAAAi+oV6AIAAAC6EmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYWmigCwgGXq9X58+fV0xMjGw2W6DLAQAA7WAYhqqrq5WQkKBevdq+fkPYkXT+/HklJiYGugwAANAJZ8+e1dChQ9tcT9iRFBMTI+nKmxUbGxvgagAAQHs4nU4lJiaan+NtIexI5q2r2NhYwg4AAD3Mtbqg0EEZAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYWsDDzrlz5/TNb35TAwYMUFRUlO644w4VFhaa6w3DUGZmphISEhQZGam0tDSdOHHCZx8ul0vLly9XXFycoqOjNXv2bJWWlnb3oQAAgCAU0LBTWVmpe+65R2FhYfrDH/6g999/Xxs3blTfvn3NNhs2bNCmTZuUnZ2tgoICORwOzZgxQ9XV1WabjIwM7d69Wzk5OTp8+LBqamo0a9YseTyeABzV9WtoaFBRUZHP1NDQEOiyAADokWyGYRiBevHHH39cf/rTn/TWW2+1ut4wDCUkJCgjI0M/+MEPJF25ihMfH6+f/vSnWrRokaqqqjRw4EC9+uqrmjdvniTp/PnzSkxM1Ouvv6777rvvmnU4nU7Z7XZVVVUpNjbWfwfYSUVFRVr87P+oz6ChkqSailK9sHSWUlNTA1wZAADBo72f3wG9srNnzx5NmDBBX//61zVo0CCNGzdOL730krm+pKRE5eXlSk9PN5dFRERoypQpys/PlyQVFhaqvr7ep01CQoJSU1PNNs25XC45nU6fKdj0GTRUfYeMVN8hI83QAwAAOi6gYef06dN6/vnnlZycrDfeeEOLFy/WY489pldeeUWSVF5eLkmKj4/32S4+Pt5cV15ervDwcPXr16/NNs2tX79edrvdnBITE/19aAAAIEgENOx4vV7deeedysrK0rhx47Ro0SI9+uijev75533a2Ww2n3nDMFosa+5qbdasWaOqqipzOnv27PUdCAAACFoBDTuDBw9WSkqKz7Jbb71VZ86ckSQ5HA5JanGFpqKiwrza43A45Ha7VVlZ2Wab5iIiIhQbG+szAQAAawpo2Lnnnnv0wQcf+Cz78MMPNXz4cElSUlKSHA6HcnNzzfVut1t5eXmaPHmyJGn8+PEKCwvzaVNWVqaioiKzDQAAuHGFBvLFv/e972ny5MnKysrS3Llz9X//93968cUX9eKLL0q6cvsqIyNDWVlZSk5OVnJysrKyshQVFaX58+dLkux2uxYuXKiVK1dqwIAB6t+/v1atWqWxY8dq+vTpgTw8AAAQBAIadu666y7t3r1ba9as0bp165SUlKQtW7boG9/4htlm9erVqqur05IlS1RZWamJEydq3759iomJMdts3rxZoaGhmjt3rurq6jRt2jRt375dISEhgTgsAAAQRAL6PTvBIhi/Z2fVb46p75CRkqTPz32kZ75+B9+zAwBAEz3ie3YAAAC6GmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYGmEHAABYWkDDTmZmpmw2m8/kcDjM9YZhKDMzUwkJCYqMjFRaWppOnDjhsw+Xy6Xly5crLi5O0dHRmj17tkpLS7v7UAAAQJAK+JWdMWPGqKyszJyOHz9urtuwYYM2bdqk7OxsFRQUyOFwaMaMGaqurjbb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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data[data['Category']==1]['num_words'])\n",
    "sns.histplot(data[data['Category']==0]['num_words'],color='r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "0e9b2da1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='num_sentences', ylabel='Count'>"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AID1CEQAAMB6BCIAAGA9AhEAALAegQgAAFiPQAQAAKxHIAIAANbzaSCaMWOGevfuraCgIIWFhWn48OHat2+fV40xRhkZGYqKilKbNm3Uv39/7dmzx6umsrJS48ePV4cOHdSuXTsNGzZMhw4d8qopKSlRWlqa3G633G630tLSVFpa2tBDBAAAzYBPA9H69ev19NNPa8uWLcrKytLp06eVnJyskydPOjWzZs3S7NmzNW/ePG3fvl0RERG6++67VV5e7tRMnDhRy5cv19KlS7Vx40adOHFCqampqqmpcWpGjRqlnJwcZWZmKjMzUzk5OUpLS2vU8QIAgCbKNCGFhYVGklm/fr0xxpgzZ86YiIgIM3PmTKfm1KlTxu12m1dffdUYY0xpaanx9/c3S5cudWoOHz5srrnmGpOZmWmMMearr74yksyWLVucms2bNxtJ5uuvv76s3jwej5FkPB7PVY/z7x0/ftwMf/ETM/zFT8zx48fr9dwAANjucv/9blJriDwejyQpJCREkpSbm6uCggIlJyc7NYGBgerXr582bdokScrOzlZ1dbVXTVRUlOLj452azZs3y+12q0+fPk5N37595Xa7nZqzVVZWqqyszGsDAAAtU5MJRMYYpaen684771R8fLwkqaCgQJIUHh7uVRseHu4cKygoUEBAgNq3b3/RmrCwsHM+MywszKk524wZM5z1Rm63W9HR0Vc3QAAA0GQ1mUA0btw47dy5U//1X/91zjGXy+X12hhzzr6znV1zvvqLnWfKlCnyeDzOlpeXdznDAAAAzVCTCETjx4/XRx99pLVr1+qGG25w9kdEREjSOVdxCgsLnatGERERqqqqUklJyUVrjh49es7nHjt27JyrT7UCAwMVHBzstQEAgJbJp4HIGKNx48Zp2bJlWrNmjWJjY72Ox8bGKiIiQllZWc6+qqoqrV+/XklJSZKknj17yt/f36smPz9fu3fvdmoSExPl8Xi0bds2p2br1q3yeDxODQAAsJefLz/86aef1pIlS/TnP/9ZQUFBzpUgt9utNm3ayOVyaeLEiZo+fbo6d+6szp07a/r06Wrbtq1GjRrl1I4ZM0aTJk1SaGioQkJCNHnyZCUkJGjw4MGSpK5du2ro0KEaO3asFixYIEl6/PHHlZqaqi5duvhm8AAAoMnwaSB65ZVXJEn9+/f32r9w4UI9+uijkqRn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jrfEeb7UmM+nuc/z+RhjVFlZ2SLn+EJqx3w+LXGOr4ivVnO3REuXLjX+/v7mzTffNF999ZWZOHGiadeunTl48KCvW2sQkyZNMuvWrTPfffed2bJli0lNTTVBQUEtarzl5eXmiy++MF988YWRZGbPnm2++OIL89e//tUYY8zMmTON2+02y5YtM7t27TIPP/ywiYyMNGVlZT7uvO4uNuby8nIzadIks2nTJpObm2vWrl1rEhMTzfXXX99sx/zrX//auN1us27dOpOfn+9sP/zwg1PT0ub5UmNuifM8ZcoUs2HDBpObm2t27txpnn32WXPNNdeYlStXGmNa3hwbc/Ext8Q5vloEonr28ssvm5iYGBMQEGB+8pOfeN3G2tI8+OCDJjIy0vj7+5uoqCgzYsQIs2fPHl+3Va/Wrl1rJJ2zjR492hjz4y3Z06ZNMxERESYwMNDcddddZteuXb5t+ipdbMw//PCDSU5ONh07djT+/v7mxhtvNKNHjzbff/+9r9uus/ONVZJZuHChU9PS5vlSY26J8/yrX/3K+W9zx44dzaBBg5wwZEzLm2NjLj7mljjHV8tljDGNdz0KAACg6WENEQAAsB6BCAAAWI9ABAAArEcgAgAA1iMQAQAA6xGIAACA9QhEAADAegQiAABgPQIRAACwHoEIAC7DokWLdN111/m6DQANhEAEAACsRyACcMX69++vCRMm6JlnnlFISIgiIiKUkZEhSTp48KBcLpdycnKc+tLSUrlcLq1bt06StG7dOrlcLn366ae6/fbb1aZNGw0cOFCFhYX65JNP1LVrVwUHB+vhhx/WDz/8cFk9/c///I8SEhLUpk0bhYaGavDgwTp58qRzfOHCheratatat26tW2+9VfPnz3eO1fa8bNkyDRgwQG3btlWPHj20efNmp99/+qd/ksfjkcvlksvlcsZbVVWlZ555Rtdff73atWunPn36OOOU/nZl6dNPP1XXrl117bXXaujQocrPz/fq/6233lL37t0VGBioyMhIjRs3zjnm8Xj0+OOPKywsTMHBwRo4cKC+/PJL5/iXX36pAQMGKCgoSMHBwerZs6d27NhxWX9vAP6Pr58uC6D56devnwkODjYZGRnmm2++MW+//bZxuVxm5cqVJjc310gyX3zxhVNfUlJiJJm1a9caY4xZu3atkWT69u1rNm7caD7//HNzyy23mH79+pnk5GTz+eefmw0bNpjQ0FAzc+bMS/Zz5MgR4+fnZ2bPnm1yc3PNzp07zcsvv2zKy8uNMca89tprJjIy0nzwwQfmu+++Mx988IEJCQkxixYtMsYYp+dbb73V/O///q/Zt2+feeCBB0xMTIyprq42lZWVZu7cuSY4ONjk5+eb/Px859yjRo0ySUlJZsOGDebAgQPm3//9301gYKD55ptvjDHGLFy40Pj7+5vBgweb7du3m+zsbNO1a1czatQop//58+eb1q1bm7lz55p9+/aZbdu2mTlz5hhjfnwK+x133GHuvfdes337dvPNN9+YSZMmmdDQUFNUVGSMMaZ79+7ml7/8pdm7d6/55ptvzH//93+bnJycq5pjwDYEIgBXrF+/fubOO+/02te7d2/zL//yL1cUiFatWuXUzJgxw0gy3377rbPviSeeMEOGDLlkP9nZ2UaSOXjw4HmPR0dHmyVLlnjt+9Of/mQSExONMX8LRG+88YZzfM+ePUaS2bt3rzHmx2Djdru9znHgwAHjcrnM4cOHvfYPGjTITJkyxXmfJHPgwAHn+Msvv2zCw8Od11FRUWbq1Knn7X316tUmODjYnDp1ymv/zTffbBYsWGCMMSYoKMgJdwDqxs8316UANHe33Xab1+vIyEgVFhbW+Rzh4eFq27atbrrpJq9927Ztu+R5evTooUGDBikhIUFDhgxRcnKyHnjgAbVv317Hjh1TXl6exowZo7FjxzrvOX36tNxu9wX7iYyMlCQVFhbq1ltvPe/nfv755zLGKC4uzmt/ZWWlQkNDnddt27bVzTff7HXu2r+rwsJCHTlyRIMGDTrvZ2RnZ+vEiRNe55OkiooKffvtt5Kk9PR0PfbYY3r33Xc1ePBg/fznP/f6PACXRiACUCf+/v5er10ul86cOaNrrvlxaaIxxjlWXV19yXO4XK4LnvNSWrVqpaysLG3atEkrV67USy+9pKlTp2rr1q1q27atJOn1119Xnz59znnfxfqRdNHPP3PmjFq1aqXs7OxzznXttdee97y15679+2nTps1Fx3bmzBlFRkZ6rUuqVXvXW0ZGhkaNGqWPP/5Yn3zyiaZNm6alS5fqvvvuu+i5AfwNgQhAverYsaMkKT8/X7fffrskeS2wbigul0t33HGH7rjjDv3hD39QTEyMli9frvT0dF1//fX67rvv9Itf/KLO5w8ICFBNTY3Xvttvv101NTUqLCzUT3/60zqdNygoSJ06ddLq1as1YMCAc47/5Cc/UUFBgfz8/NSpU6cLnicuLk5xcXH67W9/q4cfflgLFy4kEAFXgEAEoF61adNGffv21cyZM9WpUycdP35czz33XIN+5tatW7V69WolJycrLCxMW7du1bFjx9S1a1dJP15BmTBhgoKDg5WSkqLKykrt2LFDJSUlSk9Pv6zP6NSpk06cOKHVq1erR48eatu2reLi4vSLX/xCjzzyiF588UXdfvvtOn78uNasWaOEhATdc889l3XujIwMPfnkkwoLC1NKSorKy8v1l7/8RePHj9fgwYOVmJio4cOH69/+7d/UpUsXHTlyRCtWrNDw4cPVvXt3/fM//7MeeOABxcbG6tChQ9q+fbvuv//+Ov99AjbitnsA9e6tt95SdXW1evXqpd/85jd64YUXGvTzgoODtWHDBt1zzz2Ki4vTc889pxdffFEpKSmSpMcee0xvvPGGFi1apISEBPXr10+LFi1SbGzsZX9GUlKSnnzyST344IPq2LGjZs2aJenH2/kfeeQRTZo0SV26dNGwYcO0detWRUdHX/a5R48erblz52r+/Pnq3r27UlNTtX//fkk/XvlasWKF7rrrLv3qV79SXFycHnroIR08eFDh4eFq1aqVioqK9MgjjyguLk4jR45USkqK/vjHP17B3yAAl/n7L/oBAAAsxBUiAABgPQIRgCbv+++/17XXXnvB7fvvv/d1iwCaOb4yA9DknT59WgcPHrzg8U6dOsnPj3tEANQdgQgAAFiPr8wAAID1CEQAAMB6BCIAAGA9AhEAALAegQgAAFiPQAQAAKxHIAIAANb7/wFNrdOlFNfleQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data[data['Category']==1]['num_sentences'])\n",
    "sns.histplot(data[data['Category']==0]['num_sentences'],color='r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "b2245127",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.PairGrid at 0x20adb9ba190>"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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SUlJClALFy5USim1cOFCBaguXbqoZcuWqXXr1qnGjRuruLg41adPH7Vu3Tq1ZMkSFRoaql577bUytenJJ59UwcHBaurUqWrPnj1q6dKlavLkyUopVdCmZs2aqd9//13t3LlTDRw4UNWrV0/l5eUppZQ6cuSIevPNN9X69evV3r171fvvv6/MZrNauXKly3X7+fmpJ554Qu3YsUNt375dKaXUu+++qxYsWKD27dun5s+fr5o2baoeeOCBgv3Wr1+vrFareuCBB9SGDRvUli1b1AcffKBOnjypkpOTVdeuXdW9996rjh8/ro4fP67y8/PVsWPHVFhYmBo/frzavn27WrdunbrmmmtUz549S23Pjz/+qAICAtTs2bPVwYMH1apVq9Snn356Xt+zpmZS1fpvcnKyMplMas2aNUoppSZNmqTCwsJU586dC7aJjY1VH3/8sVJKqTVr1iiTyaReeukltXPnTjVlyhTl7e2tpkyZUrB9vXr1VEBAgHrzzTfV7t271e7du9XKlSuVYRhq4sSJaufOneq9995TQUFBKjAwsGC/li1bqmHDhqnt27erXbt2qR9++EFt2LDh/G+2psZR1frPsWPHlIeHh3rnnXfU/v371aZNm9SHH36o0tLSlFJKffrppyo6OlrNmDFD7du3T82YMUOFhISoqVOnKqXO/n7MyclRkyZNUgEBAQXvKMexhw4dqrp166aWLFmi9uzZo958801ltVrVrl27lFJKTZkyRVksFtW7d2+1evVqtXbtWtW8eXM1dOjQgvZ/9NFHysvLS02aNEnt3LlT/fPPP+rdd99VSillt9tV9+7dVf/+/dXq1avVrl271Lhx41RoaKhKTExUSuk+q7n4aMOmBhIXF6euuOIKl2WdO3dWTz311Dn9sP/1118F20ycOFEBau/evQXL7r//ftW3b9+ztic1NVVZrdYCQ6YojjZ99tlnBcu2bt2qgALjxB3XXXedGjdunMt1t2vX7qzt+eGHH1RoaGjB30OGDFHdu3cvcfu4uDg1ZswYl2XPPfec6tOnj8uyw4cPK0Dt3LmzxPa8/fbbKjY2VuXm5p61nZpLk6rWf5VSqkOHDuqtt95SSik1YMAA9eqrrypPT0+Vmpqqjh8/7tJXhw4dqq655hqX/Z944gnVokWLgr/r1aunBgwY4LLNkCFD1LXXXuuybPDgwS6Gjb+/f8GAT6NxR1XrP2vXrlWAOnDggNv1derUUd9++63Lspdffll17dpVKVW29+OUKVNc+olSSu3Zs0cZhqGOHj3qsvzqq69W48ePL9gPUHv27ClY/+GHH6rIyMiCv2NiYtSzzz7rtu3z589XAQEBKjs722V5o0aN1H//+1+llO6zmouPDkWrobRp08bl7+joaBISEs77GJGRkfj4+NCwYUOXZWU55vbt28nJyeHqq68u8/mio6MBCo5vs9l49dVXadOmDaGhofj5+TF37lwOHTrkcoxOnToVO+7ChQu55pprqFWrFv7+/owYMYLExMSCUIANGzactW1FWbt2LQsXLsTPz6/g06xZMwD27t1bYnsGDRpEVlYWDRs25N577+Xnn38mPz//nM6tqflUpf4LEt6zaNEilFIsXbqUm266iVatWrFs2TIWLlxIZGRkwfO/fft2unfv7rJ/9+7d2b17NzabrWBZ0b6xfft2unbt6rKs6N+PPfYYo0aNonfv3rz22msufU2jcVCV+k/btm25+uqrad26NYMGDWLy5MkkJSUBcPLkSQ4fPsw999zj8i555ZVXij3bpb0f3bFu3TqUUsTGxroce/HixS7H9vHxoVGjRi7Hdhw3ISGBY8eOlfh+XLt2Lenp6QXvZMdn//79BefQfVZzsfGo7AZoKgaLxeLyt2EY2O12TCaxZZVSBevy8vLOegzDMEo85tnw9vY+5zYbhgFQcPy3336bd999l0mTJhXky4wdO7ZY0rGvr6/L3wcPHuS6665j9OjRvPzyy4SEhLBs2TLuueeegusua/sKY7fb6d+/P6+//nqxdY6Xjrv21KlTh507dzJv3jz++usvHnzwQd58800WL15c7P5qLl2qUv8FMWw+//xzNm7ciMlkokWLFsTFxbF48WKSkpKIi4sr2FYpVdB/Cy8rStG+4W6bokyYMIGhQ4cya9Ys/vjjD1544QWmT5/OzTffXKbr0FwaVKX+YzabmTdvHitWrGDu3Ll88MEHPPvss6xatQofHx8AJk+ezOWXX15sv9LaA5R6frvdjtlsZu3atcWO5efn5/a4jmM77s/Z3o12u53o6GiXvB0HDpU23Wc1FxvtsbnECA8PB3BJDqzo+jFNmjTB29ub+fPnn/cxHLPEw4YNo23btjRs2JDdu3efdb81a9aQn5/P22+/TZcuXYiNjeXYsWMu27Rp06bUtnl6errMNAN06NCBrVu3Ur9+fRo3buzyKTpgK4q3tzc33ngj77//PosWLeLvv/9m8+bNZ70WjaYy+i/AVVddRVpaGpMmTSIuLg7DMIiLi2PRokUsWrTIxbBp0aIFy5Ytc9l/xYoVxMbGFhtgFaZFixasXLnSZVnRvwFiY2P5v//7P+bOncstt9zClClTLvDqNJcKldV/DMOge/fuvPjii6xfvx5PT09+/vlnIiMjqVWrFvv27Sv2HmnQoEGZj+/uHdW+fXtsNhsJCQnFjh0VFVWm4/r7+1O/fv0S348dOnQgPj4eDw+PYucICwsr2E73Wc3FRHtsLjG8vb3p0qULr732GvXr1+fUqVP861//qtBzenl58dRTT/Hkk0/i6elJ9+7dOXnyJFu3buWee+4p0zEaN27MjBkzWLFiBcHBwbzzzjvEx8fTvHnzUvdr1KgR+fn5fPDBB/Tv35/ly5fzySefuGwzfvx4WrduzYMPPsjo0aPx9PRk4cKFDBo0iLCwMOrXr8+qVas4cOAAfn5+hISE8NBDDzF58mSGDBnCE088QVhYGHv27GH69OlMnjy5xAHc1KlTsdlsXH755fj4+PD111/j7e1NvXr1ynYzNZc0ldF/AQIDA2nXrh3Tpk3jvffeA8TYGTRoEHl5efTo0aNg23HjxtG5c2defvllBg8ezN9//81//vMfF6Undzz66KN069aNN954gwEDBjB37lz+/PPPgvVZWVk88cQTDBw4kAYNGnDkyBFWr17NrbfeWiHXrKl5VEb/WbVqFfPnz6dPnz5ERESwatUqTp48WfDumjBhAo8++igBAQH069ePnJwc1qxZQ1JSEo899liZzlG/fn3S09OZP38+bdu2xcfHh9jYWO644w5GjBjB22+/Tfv27Tl16hQLFiygdevWXHfddWU69oQJExg9ejQRERH069ePtLQ0li9fziOPPELv3r3p2rUrAwYM4PXXX6dp06YcO3aM2bNnM2DAAFq2bKn7rOaioz02lyBffPEFeXl5dOrUiTFjxvDKK69U+Dmfe+45xo0bx/PPP0/z5s0ZPHjwOcU8P/fcc3To0IG+ffvSo0cPoqKiGDBgwFn3a9euHe+88w6vv/46rVq14ptvvmHixIku28TGxjJ37lw2btzIZZddRteuXfnf//6Hh4fY/Y8//jhms5kWLVoQHh7OoUOHiImJYfny5dhsNvr27UurVq0YM2YMgYGBBeEO7ggKCmLy5Ml07969wFP022+/ERoaWuZ7obm0qYz+C9CzZ09sNluBERMcHFzQJwpPMHTo0IEffviB6dOn06pVK55//nleeukl7rzzzlKP36VLFz777DM++OAD2rVrx9y5c10GnWazmcTEREaMGEFsbCy33XYb/fr148UXX6yIy9XUUC52/wkICGDJkiVcd911xMbG8q9//Yu3336bfv36ATBq1Cg+++wzpk6dSuvWrYmLi2Pq1Knn5LHp1q0bo0ePZvDgwYSHh/PGG28AIiM9YsQIxo0bR9OmTbnxxhtZtWoVderUKfOxR44cyaRJk/joo49o2bIlN9xwQ0G0hGEYzJ49m6uuuoq7776b2NhYbr/9dg4cOEBkZKTus5pKwVBlCWzWaDQajUaj0Wg0miqM9thoNBqNRqPRaDSaao82bDQXzKFDh1ykHot+ikoyazSaqoPuvxrN+aP7j0ZTtdChaJoLJj8/nwMHDpS4vn79+gX5KhqNpmqh+69Gc/7o/qPRVC20YaPRaDQajUaj0WiqPToUTaPRaDQajUaj0VR7tGGj0Wg0Go1Go9Foqj3asNFoNBqNRqPRaDTVHm3YaDQajUaj0Wg0mmqPNmw0Go1Go9FoNBpNtUcbNhqNRqPRaDQajabaow0bjUaj0Wg0Go1GU+3Rho1Go9FoNBqNRqOp9vw/sD0fjyTpuGIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 824.861x750 with 12 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.pairplot(data,hue='Category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "37e15d81",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ameya\\AppData\\Local\\Temp\\ipykernel_16668\\2578434383.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.\n",
      "  sns.heatmap(data.corr(),annot=True)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(data.corr(),annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e333d8e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "599608f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=data['Message']\n",
    "y=data['Category']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "d27282e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       Go until jurong point, crazy.. Available only ...\n",
       "1                           Ok lar... Joking wif u oni...\n",
       "2       Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3       U dun say so early hor... U c already then say...\n",
       "4       Nah I don't think he goes to usf, he lives aro...\n",
       "                              ...                        \n",
       "5567    This is the 2nd time we have tried 2 contact u...\n",
       "5568                Will Ì_ b going to esplanade fr home?\n",
       "5569    Pity, * was in mood for that. So...any other s...\n",
       "5570    The guy did some bitching but I acted like i'd...\n",
       "5571                           Rofl. Its true to its name\n",
       "Name: Message, Length: 5169, dtype: object"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "f326d614",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       1\n",
       "1       1\n",
       "2       0\n",
       "3       1\n",
       "4       1\n",
       "       ..\n",
       "5567    0\n",
       "5568    1\n",
       "5569    1\n",
       "5570    1\n",
       "5571    1\n",
       "Name: Category, Length: 5169, dtype: object"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "e1673855",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((5169,), (4652,), (517,))"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)\n",
    "# 10% of the data will be used for testing, and the remaining 90% will be used for training.\n",
    "x.shape,x_train.shape,x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "64eb82a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "76869f05",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_vector=tfidf.fit_transform(x_train)\n",
    "x_test_vector=tfidf.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "99e39776",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train=y_train.astype('int')\n",
    "y_test=y_test.astype('int')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "ec1656c7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2931    Only 2% students solved this CAT question in '...\n",
       "4395    Dear :-/ why you mood off. I cant drive so i b...\n",
       "3011    Yeah no probs - last night is obviously catchi...\n",
       "699              K..u also dont msg or reply to his msg..\n",
       "907     I.ll give her once i have it. Plus she said gr...\n",
       "                              ...                        \n",
       "3534                             I'm at home. Please call\n",
       "1124                   Aiyar sorry lor forgot 2 tell u...\n",
       "2628    Haha... They cant what... At the most tmr forf...\n",
       "3833           Watching tv lor. Nice one then i like lor.\n",
       "2694    Hey sexy buns! What of that day? No word from ...\n",
       "Name: Message, Length: 4652, dtype: object"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "ec069124",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 1019)\t0.2871116611569076\n",
      "  (0, 3280)\t0.13522988190511168\n",
      "  (0, 356)\t0.2871116611569076\n",
      "  (0, 1549)\t0.27381045446393654\n",
      "  (0, 1014)\t0.2210131461266699\n",
      "  (0, 6906)\t0.15390959106928848\n",
      "  (0, 3345)\t0.4101079933193608\n",
      "  (0, 4345)\t0.4101079933193608\n",
      "  (0, 7773)\t0.27381045446393654\n",
      "  (0, 5647)\t0.2169071927376325\n",
      "  (0, 1730)\t0.25705291584987466\n",
      "  (0, 6428)\t0.27381045446393654\n",
      "  (0, 6675)\t0.26437309812667403\n",
      "  (1, 1567)\t0.38415706407701217\n",
      "  (1, 2540)\t0.7553898196902703\n",
      "  (1, 4692)\t0.4446650176423577\n",
      "  (1, 2275)\t0.2899699855384223\n",
      "  (2, 6449)\t0.310556928118986\n",
      "  (2, 6491)\t0.3411300936290795\n",
      "  (2, 1733)\t0.4523150840498148\n",
      "  (2, 4982)\t0.4523150840498148\n",
      "  (2, 4875)\t0.2781941296081459\n",
      "  (2, 5551)\t0.46846142668128643\n",
      "  (2, 7810)\t0.28488366233119117\n",
      "  (3, 5841)\t0.3961586417249249\n",
      "  :\t:\n",
      "  (4646, 5724)\t0.43532564027567494\n",
      "  (4646, 4280)\t0.3792577437753325\n",
      "  (4646, 7089)\t0.40944653609546494\n",
      "  (4647, 3553)\t1.0\n",
      "  (4648, 3029)\t0.47703547294556264\n",
      "  (4648, 927)\t0.6114722988984475\n",
      "  (4648, 6459)\t0.37169336604983977\n",
      "  (4648, 4303)\t0.3554449933644003\n",
      "  (4648, 6906)\t0.36611703532910406\n",
      "  (4649, 3023)\t0.5596071216092731\n",
      "  (4649, 7068)\t0.3981130272873766\n",
      "  (4649, 3379)\t0.7268740516404921\n",
      "  (4650, 4868)\t0.39307770672002934\n",
      "  (4650, 7540)\t0.42202478458502174\n",
      "  (4650, 7224)\t0.4296070522452767\n",
      "  (4650, 4206)\t0.2875977314559965\n",
      "  (4650, 4303)\t0.632542744166888\n",
      "  (4651, 7829)\t0.4508829121577166\n",
      "  (4651, 7713)\t0.3580650273181609\n",
      "  (4651, 3501)\t0.2845532690565558\n",
      "  (4651, 6990)\t0.27589006866771876\n",
      "  (4651, 1604)\t0.4508829121577166\n",
      "  (4651, 6178)\t0.3799545528476921\n",
      "  (4651, 4698)\t0.3142594116145957\n",
      "  (4651, 2263)\t0.2549280336128331\n"
     ]
    }
   ],
   "source": [
    "print(x_train_vector)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38ad3250",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "703ca2b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LogisticRegression()\n",
    "model.fit(x_train_vector, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "69b09b10",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy on training data: 96.66809974204644\n"
     ]
    }
   ],
   "source": [
    "x_train_prediction=model.predict(x_train_vector)\n",
    "training_data_accuracy=accuracy_score(x_train_prediction,y_train)\n",
    "print(\"accuracy on training data:\",training_data_accuracy*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "320ba12a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy on testing data: 95.55125725338492\n"
     ]
    }
   ],
   "source": [
    "x_test_prediction=model.predict(x_test_vector)\n",
    "testing_data_accuracy=accuracy_score(x_test_prediction,y_test)\n",
    "print(\"accuracy on testing data:\",testing_data_accuracy*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cea98fbe",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "8400af19",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n",
      "ham mail\n"
     ]
    }
   ],
   "source": [
    "input=[\"Nah I don't think he goes to usf, he lives around here though\"]\n",
    "input_data=tfidf.transform(input)\n",
    "prediction=model.predict(input_data)\n",
    "print(prediction)\n",
    "\n",
    "if prediction[0]==1:\n",
    "    print('ham mail')\n",
    "else:\n",
    "    print('spam mail')\n",
    "\n",
    "       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6bed19b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d49677e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
