using Random
using DelimitedFiles
using DynamicalSystems
using Distances

Random.seed!(1234)

#################################################################
# Função para computar microestados (N=2) - integrada direto
#################################################################

const StatsBlock = 2
const Max_Micro = Int32(2^(StatsBlock * StatsBlock))

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
    data_size = length(Data_In[:])
    pow_vec = Int32[2^(i - 1) for i in 1:(StatsBlock * StatsBlock)]

    Stats = zeros(Float32, Max_Micro, TimesEps)
    R_All = zeros(Bool, data_size, data_size, TimesEps)

    dist = distancematrix(Data_In)
    RangeEps = collect(range(0.000001, maximum(dist) * 0.5, length = TimesEps))
    for loop_eps = 1:TimesEps
        Eps = RangeEps[loop_eps]
        R = RecurrenceMatrix(Data_In, Eps; metric = Euclidean())
        for k = 1:data_size
            R_All[k, :, loop_eps] = R[k, :]
        end
    end

    return Epsilon_Entropy_Method(R_All, Stats, pow_vec, data_size, StatsBlock, TimesEps)
end

#################################################################
# Geração da série BetaX
#################################################################

function gerar_serie_betaX(β, tamanho, transiente)
    x = rand()
    serie = zeros(Float64, tamanho)
    for i in 1:(tamanho + transiente)
        x = β * x
        while x > 1.0
            x -= 1.0
        end
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
        β = 1.99 + (classe * (5.00 / Classes))  # mesmo cálculo do Gera_Mapas_Jorge.jl
        println("Classe $classe → β = $β")

        for k in 1:N_series_por_classe
            serie = gerar_serie_betaX(β, data_size, transient)
            resultado = ComputeMicrostates(serie)
            probs = resultado[1:end-1]  # 16 probabilidades
            push!(resultados, (β=β, probs=probs))
        end
    end

    return resultados
end

#################################################################
# Rodar e salvar
#################################################################

resultados = main()

# Salvar resultados em arquivo .dat
output_file = "Data_Betax_RM_40classes_5.dat"
open(output_file, "w") do io
    for r in resultados
        linha = join([r.β; r.probs], " ")
        write(io, linha * "\n")
    end
end

println("Resultados salvos em: $output_file")