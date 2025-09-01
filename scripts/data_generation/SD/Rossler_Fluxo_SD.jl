using Random
using DifferentialEquations

# Configurações para padrões ordinais
const Number_of_Elements = 5
const Max_Patterns = Int64(2^(Number_of_Elements - 1))

if !isdefined(Main, :pow_vec2)
    const pow_vec2 = [2^(j-1) for j in 1:(Number_of_Elements-1)]
end

# Função para calcular os padrões ordinais
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

# Entropia de permutação normalizada
function Permutation_Entropy(probabilities)
    nonzero_probs = filter(p -> p > 0, probabilities)
    return -sum(p * log(p) for p in nonzero_probs) / log(length(probabilities))
end

# Definição do sistema de Rössler
function rossler!(du, u, p, t)
    a, b, c = p
    du[1] = -u[2] - u[3]
    du[2] = u[1] + a * u[2]
    du[3] = b + u[3] * (u[1] - c)
end

# Geração de uma série temporal do Rössler (usando x = sol[1,:])
function gerar_serie_rossler(a, tamanho, transiente)
    b, c = 0.2, 5.7
    u0 = [rand(), rand(), 0.0]
    h_step = 1.1
    tspan = (0.0, ((tamanho + transiente) * h_step) - h_step)
    prob = ODEProblem(rossler!, u0, tspan, [a, b, c])
    sol = solve(prob, saveat=h_step)
    return sol[1, (transiente + 1):(transiente + tamanho)]
end

# Função principal
function main_rossler_sd()
    Classes = 40
    N_series_por_classe = 100
    transient = 1000
    data_size = 1000

    resultados = []

    for classe in 1:Classes
        a = 0.20 + (classe * (0.10 / Classes))
        println("Classe $classe → a = $a")

        for k in 1:N_series_por_classe
            serie = gerar_serie_rossler(a, data_size, transient)
            P = Patterns_Calculation(serie, Max_Patterns, data_size, Number_of_Elements, pow_vec2)
            push!(resultados, (a=a, probs=P, PE=Permutation_Entropy(P)))
        end
    end

    return resultados
end

# Executar e salvar resultados
resultados = main_rossler_sd()

output_file = "Data_Rossler_SD_40classes_5.dat"
open(output_file, "w") do io
    write(io, "a\tPermutation_Entropy\tPattern_Probabilities\n")
    for r in resultados
        linha = string(r.a, '\t', r.PE, '\t', join(r.probs, ","))
        write(io, linha * "\n")
    end
end

println("Resultados salvos em: $output_file")
