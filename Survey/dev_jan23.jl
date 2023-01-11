## 10.01.22
using Revise
using Survey
using Test
####
apistrat_original = load_data("apistrat")
apistrat = copy(apistrat_original)
strat_wt = SurveyDesign(apistrat, strata=:stype, weights=:pw)
### popsize as Symbol
apistrat = copy(apistrat_original)
strat_pop = SurveyDesign(apistrat, strata=:stype, popsize=:fpc)


####################
using Revise
using Survey
using Test
apisrs_original = load_data("apisrs")
apisrs_original[!, :derived_probs] = 1 ./ apisrs_original.pw
apisrs_original[!, :derived_sampsize] = fill(200.0, size(apisrs_original, 1))
apisrs = copy(apisrs_original)
srs_design = SimpleRandomSample(apisrs; weights=:pw)
####################
# Stratified
using Revise
using Survey
using Test
apistrat_original = load_data("apistrat")
apistrat_original[!, :derived_probs] = 1 ./ apistrat_original.pw
##############################
apistrat = copy(apistrat_original)
@test_throws ErrorException StratifiedSample(apistrat,:stype; popsize=-2.83, ignorefpc=true)
@test_throws ErrorException StratifiedSample(apistrat,:stype; sampsize=-300)
@test_throws ErrorException StratifiedSample(apistrat,:stype; sampsize=-2.8, ignorefpc=true)
@test_throws ErrorException StratifiedSample(apistrat,:stype; weights=50)
@test_throws ErrorException StratifiedSample(apistrat,:stype; probs=1)

apistrat = copy(apistrat_original)
strat_pop = StratifiedSample(apistrat, :stype; popsize=:fpc)
@test strat_pop.data.probs == 1 ./ strat_pop.data.weights

apistrat = copy(apistrat_original)
strat_wt = StratifiedSample(apistrat, :stype; weights=:pw)
@test strat_wt.data.probs == 1 ./ strat_wt.data.weights

apistrat3 = copy(apistrat_original)
strat_probs = StratifiedSample(apistrat3, :stype; probs=:derived_probs)
@test strat_probs.data.probs == 1 ./ strat_probs.data.weights

###
# # 0.2 release
# * Readme main page - add glm, quantile, atleast 1 plot - Ayush
# * Testing for mean, total, by, quantile -> right now <10% codedev -> 4-5h - Iulia
# * crtl replace all `fn` to just fn - Ayush - new PR
# * documentation `index` `perfromance` `R_comparison` - 5-6h - Ayush - new PR

# # Beyond the 0.2 release
# * Variance estimation, theory paper read, jacknife - Sayantika. Shortlist the formulas that need to be there, what variables come from what computation.
# * Cluster Sampling - SurveyDesign, mean or total with correct variance closed form - Shikhar, Iulia
# * Multistage sampling, get below working in Julia - Shikhar, Iulia
# `srvydsgn <- design(id= ~PSU_ID, strata = ~HR, weight = ~WEIGHT_FOR_HR, data = hr, nest = TRUE)`

# 1. Clean up design update branch with only working elements. remove structs and functions that have not been tested, or still in dev.
#  - Make issues out of things are removed, eg. poststrafity.jl etc, so everyone remembers. add them, to milestone.
# 2. Improve inline commenting, Latex formula for mean, total estimators, source and page number where taken from.

## 
using Revise
using Survey
using Test
apisrs_original = load_data("apisrs")
apisrs_original[!, :derived_probs] = 1 ./ apisrs_original.pw
apisrs_original[!, :derived_sampsize] = fill(200.0, size(apisrs_original, 1))
apisrs = copy(apisrs_original)
srs = SimpleRandomSample(apisrs; weights=:pw)

@test quantile(:api00,srs,[0.1753,0.25,0.5,0.75,0.975])[!,2] ≈ [512.8847,544,659,752.5,905] atol = 1e-4

apistrat_original = load_data("apistrat")
apistrat_original[!, :derived_probs] = 1 ./ apistrat_original.pw
apistrat_original[!, :derived_sampsize] = apistrat_original.fpc ./ apistrat_original.pw
# base functionality
apistrat = copy(apistrat_original)
dstrat = StratifiedSample(apistrat, :stype; popsize = :fpc)

quantile(:api00,dstrat,0.5)

# Adapted from R `svyquantile` source code
n = length(v)
ii = sortperm(v)
v = v[ii]
cumw <- cumsum(weights[ii])
wi = weights[ii]
sum(weights)
pk <- sum(weights)/cumw[n - 1]

quantile!()

## 12.12.22
using Revise
using Survey
using Test
nhanes_original = load_data("nhanes")
nhanes = copy(nhanes_original)
dnhanes = SingleStageSurveyDesign(nhanes; cluster = :SDMVPSU, strata=:SDMVSTRA, weights=:WTMEC2YR)
cluster = :SDMVPSU
strata = :SDMVSTRA
using DataFrames
gdf = groupby(dnhanes.data, [cluster,strata])

unique(dnhanes.data[!,dnhanes.cluster])
unique(dnhanes.data[!,dnhanes.strata])

fill(size(data_groupedby_cluster, 1),(nrow(data),))


#############################
apiclus1_original = load_data("apiclus1")
apiclus1_original[!, :pw] = fill(757/15,(size(apiclus1_original,1),)) # Correct api mistake for pw column
##############################
# one-stage cluster sample
apiclus1 = copy(apiclus1_original)
dclus1 = OneStageClusterSample(apiclus1, :dnum, :fpc)
mean(:api00,dclus1)


apiclus1_original = load_data("apiclus1")
apiclus1_original[!, :pw] = fill(757/15,(size(apiclus1_original,1),)) # Correct api mistake for pw column
apiclus2_original = load_data("apiclus2")

##############################
# one-stage cluster sample
apiclus1 = copy(apiclus1_original)
dclus1 = OneStageClusterSample(apiclus1, :dnum, :fpc)
ClusterSample(apiclus1, :dnum, :fpc;weights=:pw)
# two-stage cluster sample
apiclus2 = copy(apiclus2_original)
dclus2 = ClusterSample(apiclus2, [:dnum,:snum] ,[:fpc1,:fpc2])
# two-stage `with replacement'
dclus2wr = ClusterSample(apiclus2, [:dnum,:snum]; weights=:pw)


apiclus1_original = load_data("apiclus1")
apiclus1_original[!, :pw] = fill(757/15,(size(apiclus1_original,1),)) # Correct api mistake for pw column

apiclus2_original = load_data("apiclus2")

######
function ClusterSample(data::AbstractDataFrame, cluster::Vector{Symbol};
    popsize::Union{Nothing,Symbol,Vector{Symbol}}=nothing,
    sampsize::Union{Nothing,Symbol,Vector{Symbol}}=nothing,
    weights::Union{Nothing,Symbol,Vector{<:Real}}=nothing
)
    # If single cluster then store as Symbol not Vector
    # if size(cluster,1) == 1
    #     cluster = cluster[1]
    # end
    # Store the iterator over each cluster, as used multiple times
    data_groupedby_cluster = groupby(data, cluster)
    # If any of weights or probs given as Symbol, find the corresponding column in `data`
    if isa(weights, Symbol)
        weights = data[!, weights] # If all good with weights column, then store it as Vector
    end
    # If sampsize given as Symbol, check all records equal 
    if isa(sampsize, Symbol)
        if isnothing(popsize) && isnothing(weights)
            error("if sampsize given, and popsize not given, then weights must given to calculate popsize")
        end
        for each_cluster in keys(data_groupedby_cluster)
            if !all(w -> w == first(data_groupedby_cluster[each_cluster][!, sampsize]), data_groupedby_cluster[each_cluster][!, sampsize])
                error("sampsize must be same for all observations within each cluster in ClusterSample")
            end
        end
        # original_sampsize_colname = copy(sampsize)
        sampsize = data[!, sampsize]
        # If sampsize column not provided in constructor call, set it as nrow of cluster
    elseif isnothing(sampsize)
        sampsize = transform(data_groupedby_cluster, nrow => :counts).counts
        #TODO, should have loop for arbitrary number of sampsize columns like popsize??
    end
    # If popsize given as Symbol or Vector, check all records equal in each cluster
    if isa(popsize, Symbol)
        if !(typeof(data[!,popsize]) <: Vector{<:Real})
            error("a given popsize column is not of numeric type")
        end
        for each_cluster in keys(data_groupedby_cluster)
            if !all(w -> w == first(data_groupedby_cluster[each_cluster][!, popsize]), data_groupedby_cluster[each_cluster][!, popsize])
                error("popsize must be same for all observations within each cluster in ClusterSample")
            end
        end
        data[!, :popsize] = data[!, popsize]
        data[!, :weights] = popsize ./ sampsize
    elseif isa(popsize, Vector{Symbol})
        for (i,eachpopsize) in enumerate(popsize)
            if !(typeof(data[!,eachpopsize]) <: Vector{<:Real})
                error("a given popsize column is not of numeric type")
            end
            for each_cluster in keys(data_groupedby_cluster)
                if !all(w -> w == first(data_groupedby_cluster[each_cluster][!, eachpopsize]), data_groupedby_cluster[each_cluster][!, eachpopsize])
                    error("popsize must be same for all observations within each cluster in ClusterSample")
                end
            end
            data[!, Symbol(:popsize,'_',string(i))] = data[!, eachpopsize]
        end
    else
        error("Incorrect type of popsize argument given")
    end

    # If popsize not given then weights must have been given to estimate popsize
    if isnothing(popsize)
        # TODO
        if !(typeof(weights) <: Vector{<:Real})
            error("`weights` must be given if `popsize` is not given")
        end
        # Estimate population size(s)
        @warn "using single-stage approximation of population sizes based on weights and sample size"
        data[!,:popsize] = sampsize .* weights
    end
    # if isnothing(weights)
        
    # end
    ## Set remaining parts of data structure
    # add columns for frequency and probability weights to `data`
    @show weights
    data[!, :weights] = weights
    data[!, :probs] = 1 ./ data[!, :weights] # Many formulae are easily defined in terms of sampling probabilties
    data[!, :sampsize] = sampsize
    new(data, cluster)
end
######

function ClusterSample(data::AbstractDataFrame, cluster::Vector{Symbol}, popsize::Vector{Symbol}; kwargs...)
    data_groupedby_cluster = groupby(data, cluster)
    if !(typeof(data[!,popsize]) <: Vector{<:Real})
        error("a given popsize column is not of numeric type")
    end
    for each_cluster in keys(data_groupedby_cluster)
        if !all(w -> w == first(data_groupedby_cluster[each_cluster][!, popsize]), data_groupedby_cluster[each_cluster][!, popsize])
            error("popsize must be same for all observations within each cluster in ClusterSample")
        end
    end
    # If any of weights or probs given as Symbol, find the corresponding column in `data`
    if isa(weights, Symbol)
        weights = data[!, weights] # If all good with weights column, then store it as Vector
    end
    data[!, :popsize] = data[!, popsize]
    data[!, :weights] = popsize ./ sampsize

    data[!, :weights] = weights
    data[!, :probs] = 1 ./ data[!, :weights] # Many formulae are easily defined in terms of sampling probabilties
    data[!, :sampsize] = transform(data_groupedby_cluster, nrow => :counts).counts
    new(data, cluster, popsize, sampsize, false, false)
end

function ClusterSample(data::AbstractDataFrame, cluster::Union{Symbol,Vector{Symbol}}, weights::Symbol)
    # Two stage `with replacement` like R
    new(data, cluster, nothing, sampsize, false, false)
end
