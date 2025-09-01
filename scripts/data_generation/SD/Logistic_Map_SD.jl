using Random

#################################################################
# Cálculo de padrões ordinais e entropia de permutação
#################################################################

const Number_of_Elements = 5
const Max_Patterns = Int64(2^(Number_of_Elements - 1))
const pow_vec2 = [2^(j-1) for j in 1:(Number_of_Elements-1)]

function Patterns_Calculation(Data, Max_Patterns, Window_Size, Number_of_Elements, pow_vec2)
    Stats = zeros(Float64, Max_Patterns)
    for i in 1:(Window_Size - Number_of_Elements)
        Add = 0
        for j in 1:(Number_of_Elements-1)
            a_binary = (Data[i+j-1] > Data[i+j]) ? 1 : 0
            Add += a_binary * pow_vec2[j]
        end
        Stats[Int64(Add) + 1] += 1
    end
    return Stats / sum(Stats)
end

function Permutation_Entropy(probabilities)
    nonzero_probs = filter(p -> p > 0, probabilities)
    return -sum(p * log(p) for p in nonzero_probs) / log(length(probabilities))
end

#################################################################
# Geração da série Logistic
#################################################################

function gerar_serie_logistic(r, tamanho, transiente)
    x = rand()
    serie = zeros(Float64, tamanho)
    for i in 1:(tamanho + transiente)
        x = r * x * (1.0 - x)
        if i > transiente
            serie[i - transiente] = x
        end
    end
    return serie
end

#################################################################
# Execução principal
#################################################################

function main()
    Classes = 40
    N_series_por_classe = 100
    transient = 1000
    data_size = 1000

    resultados = []

    for classe in 1:Classes
        r = 3.60 + (classe * (0.40 / Classes))
        println("Classe $classe → r = $r")

        for k in 1:N_series_por_classe
            serie = gerar_serie_logistic(r, data_size, transient)
            P = Patterns_Calculation(serie, Max_Patterns, data_size, Number_of_Elements, pow_vec2)
            push!(resultados, (r=r, probs=P, PE=Permutation_Entropy(P)))
        end
    end

    return resultados
end

#################################################################
# Rodar e salvar
#################################################################

resultados = main()

output_file = "Data_Logistic_SD_40classes_5.dat"
open(output_file, "w") do io
    write(io, "r\tPermutation_Entropy\tPattern_Probabilities\n")
    for r in resultados
        linha = string(r.r, '\t', r.PE, '\t', join(r.probs, ","))
        write(io, linha * "\n")
    end
end

println("Resultados salvos em: $output_file")