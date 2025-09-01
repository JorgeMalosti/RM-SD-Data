using Random
using DifferentialEquations

# Parâmetros para padrões ordinais
const Number_of_Elements = 5
const Max_Patterns = Int64(2^(Number_of_Elements - 1))
const pow_vec2 = [2^(j-1) for j in 1:(Number_of_Elements-1)]

# Função para calcular padrões ordinais
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

# Entropia de permutação
function Permutation_Entropy(probabilities)
    nonzero_probs = filter(p -> p > 0, probabilities)
    return -sum(p * log(p) for p in nonzero_probs) / log(length(probabilities))
end

# Definição do sistema de Lorenz
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Gerar série do sistema de Lorenz (coordenada x)
function gerar_serie_lorenz(ρ, tamanho, transiente)
    σ = 10.0
    β = 8 / 3
    u0 = [rand(), rand(), 0.0]
    h_step = 0.25
    tspan = (0.0, ((tamanho + transiente) * h_step) - h_step)
    prob = ODEProblem(lorenz!, u0, tspan, [σ, ρ, β])
    sol = solve(prob, saveat=h_step)
    return sol[1, (transiente + 1):(transiente + tamanho)]
end

# Execução principal
function main()
    Classes = 40
    N_series_por_classe = 100
    transient = 1000
    data_size = 1000

    resultados = []

    for classe in 1:Classes
        ρ = 27.99 + (classe * (10.0 / Classes))
        println("Classe $classe → ρ = $ρ")

        for k in 1:N_series_por_classe
            serie = gerar_serie_lorenz(ρ, data_size, transient)
            P = Patterns_Calculation(serie, Max_Patterns, data_size, Number_of_Elements, pow_vec2)
            push!(resultados, (ρ=ρ, probs=P, PE=Permutation_Entropy(P)))
        end
    end

    return resultados
end

# Rodar e salvar
resultados = main()

output_file = "Data_Lorenz_SD_40classes_5.dat"
open(output_file, "w") do io
    write(io, "ρ\tPermutation_Entropy\tPattern_Probabilities\n")
    for r in resultados
        linha = string(r.ρ, '\t', r.PE, '\t', join(r.probs, ","))
        write(io, linha * "\n")
    end
end

println("Resultados salvos em: $output_file")
