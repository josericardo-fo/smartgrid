# Smartgrid
### Gerenciamento de Roteamento e Direcionamento de Eletricidade em Estações de Carregamento de Carros Elétricos

## Descrição do Projeto

Este projeto tem como objetivo gerenciar o roteamento e o direcionamento de eletricidade em vagas de estacionamento com carregamento de carros elétricos em horários de pico, sem a necessidade de expansão do transformador. Utilizando grafos para modelar e otimizar o carregamento, garantimos a eficiência na distribuição de energia e priorizamos carros com menor nível de bateria.

## Motivação

Com o aumento do número de veículos elétricos, o carregamento residencial se torna crucial, representando 80% de todo o carregamento feito por motoristas de veículos elétricos. A infraestrutura de carregamento deve ser eficiente e capaz de atender à demanda sem sobrecarregar a rede elétrica existente. Este projeto aborda esses desafios utilizando técnicas de grafos e algoritmos de fluxo máximo.

## Funcionalidades

1. **Distribuição de Energia**: O sistema utiliza três carregadores (Wallbox), cada um suportando três carros. A distribuição é feita de forma equilibrada entre os carregadores.
2. **Prioridade de Carregamento**: Carros com baterias abaixo de 80% carregam mais rapidamente, enquanto carros com baterias acima de 80% carregam mais lentamente.
3. **Visualização em Tempo Real**: O projeto utiliza Dash para visualizar o estado do carregamento em tempo real, atualizando a cada segundo.
4. **Indicadores Visuais**:
   - Carros com baterias totalmente carregadas (100%) são exibidos na cor verde.
   - Carros com baterias acima de 80% (carregando lentamente) são exibidos na cor laranja.
   - Carros com baterias abaixo de 80% são exibidos na cor azul.

## Como Funciona

### Estrutura do Grafo

- **Nós**: Representam os carregadores e os carros.
- **Arestas**: Representam a capacidade de carregamento entre o transformador e os carregadores, e entre os carregadores e os carros.
- **Capacidade**: Carros com baterias abaixo de 80% têm uma capacidade de carregamento de 7.4 kW, enquanto carros acima de 80% têm uma capacidade de 2.0 kW.

### Algoritmo de Carregamento

1. **Atualização Periódica**: A cada segundo, o sistema atualiza o nível de bateria dos carros.
2. **Lógica de Carregamento**:
   - Carros com baterias abaixo de 80% aumentam 1% a cada 2 segundos.
   - Carros com baterias acima de 80% aumentam 1% a cada 4 segundos.
3. **Visualização**: A interface gráfica mostra o estado atual do carregamento e o tempo decorrido desde o início da simulação.

## Execução do Projeto

### Requisitos

- Python 3.x
- Bibliotecas: `dash`, `networkx`, `plotly`

### Instalação

1. Clone o repositório:
   ```sh
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_DIRETORIO>

2. Instale as dependências:
   ```sh
   pip install dash networkx plotly

### Execução

1. Inicie o servidor:
   ```sh
   python main.py

2. Acesse o endereço `http://127.0.0.1:8050/` no navegador.

### Código Principal

O código principal está localizado no arquivo `main.py` e contém a lógica de carregamento, atualização do grafo e a interface gráfica.

## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões, melhorias ou encontrar algum problema, sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo `LICENSE` para mais detalhes.

Obrigado por conferir nosso projeto! Esperamos que esta solução ajude a otimizar o carregamento de veículos elétricos de forma eficiente e sustentável.