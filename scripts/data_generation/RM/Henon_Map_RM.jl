using Random
using DelimitedFiles
using DynamicalSystems
using Distances

Random.seed!(1234)

#################################################################
# Parâmetros globais
#################################################################

const StatsBlock = 2
const Max_Micro = Int32(2^(StatsBlock * StatsBlock))

#################################################################
# Função personalizada de matriz de recorrência (sem RecurrenceAnalysis)
#################################################################

function manual_recurrence_matrix(data::Matrix{Float64}, ε::Float64)
    D = pairwise(Euclidean(), data, dims=2)
    return D .< ε
end

#################################################################
# Função para computar microestados (N=2)
#################################################################

function Epsilon_Entropy_Method(R, Stats, pow_vec, Size_Data, StatsBlock, TimesEps)
    Threads.@threads for IndexComputing = 1:TimesEps
        Norm_All = ((Size_Data - (StatsBlock - 1)) * (Size_Data - (StatsBlock - 1)))
        for count_X_All = 1:(Size_Data - (StatsBlock - 1))
            for count_Y_All = 1:(Size_Data - (StatsBlock - 1))
                Add = 0
                for count_x = 1:StatsBlock
                    for count_y = 1:StatsBlock
                        a_binary = R[count_y + count_Y_All - 1, count_x + count_X_All - 1, IndexComputing] ? 1 : 0
                        Add += a_binary * pow_vec[count_y + ((count_x - 1) * StatsBlock)]
                    end
                end
                Stats[Add + 1, IndexComputing] += 1
            end
        end
        Stats[:, IndexComputing] ./= Norm_All
    end

    Entropy = 0.0
    StatsMaxEnt = zeros(length(Stats[:, 1]))
    for IndexComputing = 1:TimesEps
        SEps = -sum(Stats[j, IndexComputing] * log(Stats[j, IndexComputing]) for j in 1:length(Stats[:, IndexComputing]) if Stats[j, IndexComputing] > 0)
        if SEps > Entropy
            Entropy = SEps
            StatsMaxEnt .= Stats[:, IndexComputing]
        end
    end

    return [StatsMaxEnt; Entropy]
end

function ComputeMicrostates(Data_In)
    TimesEps = 30
    data_size = size(Data_In, 2)  # matriz (1, N)
    pow_vec = Int32[2^(i - 1) for i in 1:(StatsBlock * StatsBlock)]

    Stats = zeros(Float32, Max_Micro, TimesEps)
    R_All = zeros(Bool, data_size, data_size, TimesEps)

    dist = distancematrix(Data_In)
    max_dist = maximum(dist[.!isnan.(dist)])
    RangeEps = collect(range(0.000001, max_dist * 0.5, length = TimesEps))

    for loop_eps = 1:TimesEps
        Eps = RangeEps[loop_eps]
        R = manual_recurrence_matrix(Data_In, Eps)
        R_All[:, :, loop_eps] .= R
    end

    return Epsilon_Entropy_Method(R_All, Stats, pow_vec, data_size, StatsBlock, TimesEps)
end

#################################################################
# Geração da série Henon Map
#################################################################

function gerar_serie_henon(a, b, tamanho, transiente)
    x = rand()
    y = rand()
    serie = zeros(Float64, tamanho)
    for i in 1:(transiente + tamanho)
        x_new = y + 1.0 - a * x^2
        y = b * x
        x = x_new
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

    b = 0.3
    resultados = []

    for classe in 1:Classes
        a = 1.10 + (classe * (0.10 / Classes))  # mesmo cálculo do Gera_Mapas_Jorge.jl
        println("Classe $classe → a = $a")

        for k in 1:N_series_por_classe
            serie = gerar_serie_henon(a, b, data_size, transient)
            resultado = ComputeMicrostates(reshape(serie, 1, :))  # transforma em matriz (1, N)
            probs = resultado[1:end-1]  # 16 probabilidades
            push!(resultados, (a=a, probs=probs))
        end
    end

    return resultados
end

#################################################################
# Rodar e salvar
#################################################################

resultados = main()

# Salvar resultados em arquivo .dat
output_file = "Data_Henon_RM_40classes_1.dat"
open(output_file, "w") do io
    for r in resultados
        linha = join([r.a; r.probs], " ")
        write(io, linha * "\n")
    end
end

println("Resultados salvos em: $output_file")
