import os
import csv 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Function to rename multiple files

df = pd.DataFrame(columns=['biblioteca','latent_dimens', 'cpu','memoria','tempo inferencia'])
df_files = pd.DataFrame(columns=['biblioteca', 'enc/dec','dim', 'size'])
dados = []


def read_text_file(file_path,model):
    lines = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
            	#print(line)
            	line_split=line.strip().split("//")
            	cpu=float(line_split[1])
            	memoria=float(line_split[2])/ 1024 
            	tempo_inf=int(line_split[3])
            	print(model)
            	#print(file_path, cpu,memoria,tempo_inf)
            	df.loc[len(df)] =[model.split("_")[0], float(model.split("_")[1]),cpu,memoria,tempo_inf]
            	#dados.append({'cpu': cpu, 'memoria': memoria,'tempo inferencia':tempo_inf}, ignore_index=True)
            	#lines.append(tempo_inf)
    except Exception as e:
        print('Erro ao ler o arquivo:', e)

    return df



def main():
   
    folder = "results"
    for count, filename in enumerate(os.listdir(folder)):
    	src =f"{folder}/{filename}"
    	df=read_text_file(src,filename.split('.')[0])
    	print(src)
    	#print(df)
    	
    	
    	# Criar o DataFrame a partir da lista de dados
	
	
    	
def get_size_file(folder):
	
	for count, filename in enumerate(os.listdir(folder)):
		src =f"{folder}/{filename}"
		dados_model=filename.split('_')
		tamanho_arquivo = os.path.getsize(src)		 
		df_files.loc[len(df_files)] =[dados_model[1],dados_model[2], dados_model[3].split('.')[0],tamanho_arquivo] 
		
	 	

def plot_size(df):
	df=df.sort_values(by=['Modelo'])
	plt.figure(figsize=(14, 5))
	g=sns.barplot(data=df,
            y=df.index,
            x="memoria",
            hue="Modelo",
            #order = [ "quant"],
            palette = "muted"
           )
           
    
    

	plt.title("Bar plot")
	plt.show()        
        
def plot_(df1):
	#sns.set_style("darkgrid")
	#sns.boxplot(data=df, x='latent_dimens', y='memoria', hue='biblioteca')
	sns.stripplot(data=df1, y="size", x="dim", hue="biblioteca", dodge=True, jitter=False,order = [ "8", "16", "32"])
	#sns.stripplot(x="latent_dimens", y="tempo inferencia", hue="biblioteca", data=df,  order = [ "8", "16", "32"],)
	plt.title("tamanho dos modelos gerados para o decoder")
	plt.ylabel('size Bytes')
	plt.show() 
	


 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    #main()
    folder = "pytorch/android_pytorch/app/src/main/assets"
    get_size_file(folder)
    foldertf = "tensorflow/android_tensorflow/app/src/main/assets"
    get_size_file(foldertf)
    print(df_files)
    
    
    #df = (df_files.loc[(df_files.biblioteca == "tensorflow") ]).sort_values(by=['size'])
    #print(newdf)
    df_files = df_files.loc[(df_files['enc/dec'] == "decoder") ]
    plot_(df_files)
    #plot_size(df)
    #plot_(df2)
    
    
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dados de exemplo
dados = {
    'biblioteca': ['tensorflow', 'tensorflow', 'tensorflow', 'tensorflow', 'tensorflow',
                   'pytorch', 'pytorch', 'pytorch', 'pytorch', 'pytorch'],
    'latent_dimens': [16, 16, 16, 16, 16, 8, 8, 8, 8, 8],
    'tempo_inferencia': [72, 50, 47, 52, 46, 16.746408, 16.480262, 17.792085, 22.035627, 3.768647]
}

# Criar DataFrame
#df = pd.DataFrame(dados)

# Criação do gráfico de boxplot usando Seaborn
#plt.figure(figsize=(10, 6))
#sns.boxplot(data=df, x='latent_dimens', y='tempo inferencia', hue='biblioteca')

plt.xlabel('Latent Dimens')
plt.ylabel('tempo usado para fazer inferencia em milissegundos')
#plt.title('Comparação do Tempo de Inferência entre TensorFlow e PyTorch')

plt.legend(title='Biblioteca')

# Exibe o gráfico
#plt.show()  
    
    

