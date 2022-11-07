# ### Lumley Texbook code, Fig 2.2 pg 20
using Revise
using Survey
using DataFrames
using CSV

# Load in dataframe
apisrs = CSV.read("assets/apisrs.csv",DataFrame)

### Set design (All should give identical results)
srs_design = SimpleRandomSample(apisrs, popsize = apisrs.fpc)     # popsize only
srs_design = SimpleRandomSample(apisrs, weights = apisrs.pw)      # no popsize, so weights given as Vector
srs_design = SimpleRandomSample(apisrs, weights = :pw)            # no popsize, so weights given as Symbol
srs_design = SimpleRandomSample(apisrs, probs = 1 ./ apisrs.pw)   # no popsize, so probs given as Vector

svytotal(:enroll,srs_design)
svymean([:enroll,:api00],srs_design)
svymean(:enroll,srs_design)

# svytotal error
svytotal(:api00, srs)

# No fpc example
no_fpc = SimpleRandomSample(apisrs, ignorefpc = true) 
svytotal(:enroll,no_fpc)
svytotal(:api00,no_fpc)
svymean(:enroll,no_fpc)

####
using Revise
using Survey
using DataFrames
using CSV
using CategoricalArrays
# Test feature for categorical variables
apisrs_categ = CSV.read("assets/apisrs.csv",DataFrame)
eltype(apisrs_categ.stype)
# Convert a column to CategoricalArray
apisrs_categ.stype = CategoricalArray(apisrs_categ.stype)
eltype(apisrs_categ.stype)

srs_design_categ = SimpleRandomSample(apisrs_categ, popsize = apisrs_categ.fpc)

# isa(srs_design_categ.data.stype, CategoricalArray)
# isa(srs_design_categ.data[!,:stype], CategoricalArray)

# Svymean and svytotal example
svymean(:enroll,srs_design_categ) # works
svymean(:stype,srs_design_categ) # no method matching /(::CategoricalValue{String1, UInt32}, ::Int64)
svytotal(:stype,srs_design_categ)

# way to update 
srs_design.data.apidiff = srs_design.data.api00 - srs_design.data.api99


svyquantile(:enroll, srs_design_categ,0.5)

# isa(srs_design_categ.data.stype, CategoricalArray)


# # apisrs = DataFrame(CSV.file("data/apisrs.csv"))
# # Base.format_bytes(Base.summarysize(apisrs.stype))
# # Base.format_bytes(Base.summarysize(CategoricalArray(apisrs.stype)))


# ### Test 10.09.22

# gdf = groupby(design.data, by)
# combine(gdf, [formula, :weights] => ((a, b) -> func(a, design, b, params...)) => AsTable)

# using Revise
# using Survey
# using DataFrames
# using CSV
# using StatsBase

# apisrs_categ = CSV.read("assets/apisrs.csv",DataFrame) # laod data
# srs_design = SimpleRandomSample(apisrs_categ, popsize = apisrs_categ.fpc) # create design object
# # manually grouby to get result
# gdf = groupby(srs_design.data, :cname )
# combine(gdf, :api00 => mean) # works
# combine(gdf, (:api00,srs_design) => svymean)

# combine(gdf, [:api00, :pw] => ((a, b) -> svymean(a, srs_design, b)) => AsTable)

# Test 12.09.22
using Revise
using Survey
using DataFrames
using CSV
using StatsBase
apisrs_categ = CSV.read("assets/apisrs.csv",DataFrame) # laod data
srs_design = SimpleRandomSample(apisrs_categ, popsize = apisrs_categ.fpc) # create design object
gdf = groupby(srs_design.data, :cname )
combine(gdf, [:api00, :pw] => ((a, b) -> svymean(a, srs_design, b)) => AsTable)



        
        # # print("Yolo")
        # test = combine(gdf, x => mean => :mean) # |> DataFrame |> AsTable # , (x , design) => sem => :sem ) |> DataFrame
        # @show test
        # # show(test)
        # # delay(50000)
        # return 0

##  21.09.22 Stratified test 1
# Ideally you should stratify on a CategoricalArray, alternatively, convert the StringX to categorical value before running StratifiedSample
using Revise
using Survey
using DataFrames
using CSV
using StatsBase
using CategoricalArrays

apistrat_categ = CSV.read("assets/apistrat.csv",DataFrame) # load data
apistrat_categ.stype = CategoricalArray(apistrat_categ.stype)
eltype(apistrat_categ.stype)

strat_categ_design = StratifiedSample(apistrat_categ, :stype ; popsize = apistrat_categ.fpc )
strat_categ_design = StratifiedSample(apistrat_categ, :stype ; weights = :pw )
svymean(:stype,strat_categ_design)
svytotal(:stype,strat_categ_design)

### Strat normal
using Revise
using Survey
using DataFrames
using CSV
using StatsBase

apistrat = CSV.read("assets/apistrat.csv",DataFrame) # laod data
strat_design = StratifiedSample(apistrat, :stype ; popsize = apistrat.fpc )
svytotal(:api00,strat_design)
svymean(:api00,strat_design)

svytotal(:enroll,strat_design)
svymean(:enroll,strat_design)

# Support for categorical var

# Test feature for categorical variables


srs_design_categ = SimpleRandomSample(apisrs_categ, popsize = apisrs_categ.fpc)

# V̂ȳₕ = Nₕ .^2 ./ nₕ .* (1 .- fₕ) .* s²ₕ 
    # V̂Ȳ̂ = 1 ./ sum(Nₕ) .* sum( Nₕ .^2 .* V̂ȳₕ)   #(Nₕ .^ 2) .* design.fpc .* s²h ./ design.sampsize     # sum(combine(gdf, [x,:weights] => ( (a,b) -> wsum(a,b) ) => :total).total)


StratifiedSample(apistrat, :stype ; weights = :pw )


## 26.09.22 HT test
using Revise
using Survey
using DataFrames
using CSV

# Load in dataframe
apisrs = CSV.read("assets/apisrs.csv",DataFrame)

### Set design (All should give identical results)
srs_design = SimpleRandomSample(apisrs, popsize = apisrs.fpc)     # popsize only

ht_calc(:api00, srs_design)


ht_calc(:api00, strat_design)

### 17.10.22 HT svy mean total
using Revise
using Survey
using DataFrames
using CSV

# Load in dataframe
# apisrs = load_data("apisrs")
apisrs = CSV.read("assets/apisrs.csv",DataFrame)

### Set design (All should give identical results)
srs_design = SimpleRandomSample(apisrs, popsize = apisrs.fpc)
general_design = SurveyDesign(apisrs; popsize = apisrs.fpc)
general_design = SurveyDesign(apisrs; popsize = :fpc)

ht_svytotal(:api00,general_design)
svytotal(:api00,srs_design)
ht_svymean(:api00,general_design)

###
srs = SimpleRandomSample(apisrs, popsize = -2.8, ignorefpc = true)# the errror is wrong
srs = SimpleRandomSample(apisrs, sampsize = -2.8, ignorefpc = true)# the function is working upto line 55

# Domain stratified
using Revise
using Survey
using DataFrames
using StatsBase

apistrat = load_data("apistrat") # load data

strat = StratifiedSample(apistrat, :stype ; popsize = apistrat.fpc )

gdf_strata = groupby(strat.data, strat.strata)
Nₕ = combine(gdf_strata , :weights => sum => :Nₕ).Nₕ
nₕ = combine(gdf_strata, nrow => :nₕ).nₕ

# Only need nsdh and sigma_sdh_yk
gdf_domain = groupby(strat.data, domain)
domain = :cname
formula = :api00
domain_means = []
for each_domain in keys(gdf_domain)
    grouped_frame = groupby(gdf_domain[each_domain],strat.strata)
    ############ How to get 0 as nrow of empty groupedframe
    # nsdh = combine(grouped_frame, nrow=>:nsdh).nsdh # Lol this is not always length H!! sometimes strata empty in a domain
    nsdh = combine(grouped_frame, :weights => length => :nsdh).nsdh
    ############
    @show nsdh
    substrata_domain_totals = combine(grouped_frame, formula => sum => :sigma_sdh_yk).sigma_sdh_yk
    @show substrata_domain_totals, nsdh

    domain_mean_estimator = sum(Nₕ ./ nₕ .* substrata_domain_totals) / sum(Nₕ ./ nₕ .* nsdh)
    push!(domain_means,domain_mean_estimator)
end


grouped_frame = groupby(gdf_domain[(cname = "Los Angeles",)],strat.strata)



m_stype = gdf_strata[("M",)]

combine(groupby(m_stype , domain), :api00 => sum => :sigma_sd_yk).:sigma_sd_yk

combine(sum(:api00,) ) # of orange country?

gdf_strat_domain = groupby(strat.data, [strat.strata,domain])
nsdh  = combine(gdf_strat_domain,nrow => :nsdh)

# gdf_strat_domain[("E",)]
# gdf_strat_domain %> select only stype = E 

# gdf_strat_domain %> select only stype = P

for (key, subdf) in pairs(groupby(strat.data, [strat.strata,domain]))
    println("Number of data points for $(key): $(nrow(subdf))")
end



