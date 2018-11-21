import matplotlib
matplotlib.use('agg')# Configure matplotlib for non-interactive backend use on AWS
from matplotlib.pylab import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.regression import LinearRegression
from pyspark import SparkContext, SparkConf, SQLContext
from scipy import optimize

# Global Constants
# ------------------------------------------------------------------------------------------------

# --- Daily Fitting ---

# Irradiance
IRR_FIT_RANGE = [0.3, 0.7]  # Day fraction fit range for aod
VAR_THRESHOLD_IRR = 0.05    # Threshold corresponding to cloudy days
FIT_FUNCTION = lambda p, x: p[0] * np.cos(p[1] * (x - p[2]))
ERROR_FUNCTION = lambda p, x, y: FIT_FUNCTION(p, x) - y
P0_DNBI = [900, 6.0, 0.1]

# Direct-to-Diffuse Irradiance
MIN_FIT_POINTS_DDRF = 1000

# Variability
VAR_FIT_RANGE =  [0.3,0.7]  # Day fraction fit range for surface pressure
MIN_FIT_POINTS_VAR = 1000   # min points per day to fit the variability

# Feature Table
FEAT_TABLE_TOTAL_COLS = 39

# --- Seasonal Fitting ---

# Data Frames
DAY_COL_TITLE = "day"
LABEL_COL_TITLE = "label"
FEATURE_COL_TITLE = "features"
TRAIN_RANGE_DAYS = [30, 60]

# Plotting
X_AXIS_LABEL = 'Test Mid-point [Days]'
Y_AXIS_LABEL = 'Residual Sum of Squares'

# Helpers
# ------------------------------------------------------------------------------------------------

# --- All Days ---
def parse_day_line(line):
    fields = line.split(',')
    return float(fields[0])

# --- Irradiance Fitting ---
# Line : iDayDif(0), fDayDif(1), vardif(2), dif(3)
def irr_parse_line(line):
    fields = line.split(',')
    return [float(fields[0]),
            float(fields[1]),
            float(fields[2]),
            float(fields[3])]

def irr_filter_sunny_skies(line):
    # Line : iDayDif(0), fDayDif(1), vardif(2), dif(3)
    fDayDif = line[1]
    vardif = line[2]
    within_irr_range = fDayDif >= IRR_FIT_RANGE[0] and fDayDif <= IRR_FIT_RANGE[1]
    below_threshold = vardif <= VAR_THRESHOLD_IRR
    return within_irr_range and below_threshold

def irr_filter_is_cosineable(line):
    day = line[0]
    fit_day_dif = line[1][0]
    fit_diff = line[1][1]
    fit_day_dif_np = np.array(fit_day_dif)
    fit_diff_np = np.array(fit_diff)
    coefficients, success = optimize.leastsq(ERROR_FUNCTION, P0_DNBI[:], args=(fit_day_dif_np, fit_diff_np))
    return success == 1

def irr_get_cos_params_mse(line):
    day = line[0]
    fit_day_dif = line[1][0]
    fit_diff = line[1][1]
    fit_day_dif_np = np.array(fit_day_dif)
    fit_diff_np = np.array(fit_diff)

    # Collect fitting coefficients
    coefficients = optimize.leastsq(ERROR_FUNCTION, P0_DNBI[:], args=(fit_day_dif_np, fit_diff_np))

    # Collect the mean squared error
    err = 0.0
    for x, ydat in zip(fit_day_dif, fit_diff):
        erri = ydat - FIT_FUNCTION(coefficients[0], x)
        err = err + erri * erri
    mse = 0.0
    if err > 0.0 and len(fit_diff) > 0:
        mse = math.sqrt(err) / len(fit_diff)

    return day, [coefficients[0][0], coefficients[0][1], coefficients[0][2], mse]

# --- Direct-to-Diffuse Irradiance Fitting ---
def dir_to_dif_irr_parse_line(line):
    # Line : iDayDdrf (0), fDayDdrf (1), ddrf (2), varddrf (3)
    fields = line.split(',')
    return [float(fields[0]),
            float(fields[1]),
            float(fields[2]),
            float(fields[3])]

def dir_to_dif_stable_skies(line):
    # Line : iDayDdrf (0), fDayDdrf (1), ddrf (2), varddrf (3)
    fDayDdrf = line[1]
    varddrf = line[3]
    withinRange = fDayDdrf >= IRR_FIT_RANGE[0] and fDayDdrf <= IRR_FIT_RANGE[1]
    belowThreshold = varddrf <= VAR_THRESHOLD_IRR
    return withinRange and belowThreshold

def dir_to_dif_get_day_ddrf_pair(line):
    # Line : iDayDdrf (0), fDayDdrf (1), ddrf (2), varddrf (3)
    iDayDdrf = line[0]
    ddrf = line[2]
    return iDayDdrf, ddrf

def dir_to_dif_min_meas_thresh(line):
    ddrf_list = line[1]
    return len(ddrf_list) >= MIN_FIT_POINTS_DDRF

def dir_to_dif_get_avg(line):
    iDayDdrf = line[0]
    ddrf_sum = sum(line[1])
    return iDayDdrf, ddrf_sum/len(list(line[1]))

def dir_to_dif_get_std_dev(line):
    iDayDdrf = line[0]
    ddrf_data_np = np.array(line[1])
    std_dev = np.std(ddrf_data_np)
    return iDayDdrf, std_dev

# --- Variability Fitting ---
def var_parse_line(line):
    # Line : iDay (0), fDay (1), vari (2)
    fields = line.split(',')
    return [float(fields[0]),
            float(fields[1]),
            float(fields[2])]

def var_filter_clear_skies(line):
    # Line : iDay (0), fDay (1), vari (2)
    fDay = line[1]
    vari = line[2]
    variLessThanOne = vari < 1.0
    withinFitRange = fDay >= VAR_FIT_RANGE[0] and fDay <= VAR_FIT_RANGE[1]
    return variLessThanOne and withinFitRange

def var_get_day_vari_pair(line):
    # Line : iDay (0), fDay (1), vari (2)
    iDay = line[0]
    vari = line[2]
    return iDay, vari

def var_get_avg(line):
    iDay = line[0]
    vari_sum = sum(line[1])
    vari_len = len(line[1])
    return iDay, vari_sum/vari_len

def var_get_std_dev(line):
    iDay = line[0]
    vari_data_np = np.array(line[1])
    std_dev = np.std(vari_data_np)
    return iDay, std_dev

def var_filter_min_meas_thresh(line):
    vari_list = line[1]
    return len(vari_list) >= MIN_FIT_POINTS_VAR

# --- Miscellaneous ---
def day_append_beginning(line):
    day = line[0]
    day_measurements = line[1]
    float_measurements = []
    for meas in day_measurements:
        float_measurements.append(float(meas))
    return [float(day)] + float_measurements

# Initialize Spark session and collect data
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    sc = SparkContext()
    sqlContext = SQLContext(sc)

    # All days
    days_all = sc.textFile('s3://..../all_days.csv')# *** CHANGE TO CORRECT S3 PATH ***

    # Line : iDayDif(0), fDayDif(1), vardif(2), dif(3)
    irr_filter1_raw = sc.textFile('s3://..../irradiance_data_filter1.csv')# *** CHANGE TO CORRECT S3 PATH ***
    irr_filter2_raw = sc.textFile('s3://..../irradiance_data_filter2.csv')# *** CHANGE TO CORRECT S3 PATH ***
    irr_filter3_raw = sc.textFile('s3://..../irradiance_data_filter3.csv')# *** CHANGE TO CORRECT S3 PATH ***
    irr_filter4_raw = sc.textFile('s3://..../irradiance_data_filter4.csv')# *** CHANGE TO CORRECT S3 PATH ***
    irr_filter5_raw = sc.textFile('s3://..../irradiance_data_filter5.csv')# *** CHANGE TO CORRECT S3 PATH ***
    irr_filter6_raw = sc.textFile('s3://..../irradiance_data_filter6.csv')# *** CHANGE TO CORRECT S3 PATH ***

    # Line : iDayDdrf (0), fDayDdrf (1), ddrf (2), varddrf (3)
    dir_to_dif_filter1_raw = sc.textFile('s3://..../direct_to_diff_data_filter1.csv')# *** CHANGE TO CORRECT S3 PATH ***
    dir_to_dif_filter2_raw = sc.textFile('s3://..../direct_to_diff_data_filter2.csv')# *** CHANGE TO CORRECT S3 PATH ***
    dir_to_dif_filter3_raw = sc.textFile('s3://..../direct_to_diff_data_filter3.csv')# *** CHANGE TO CORRECT S3 PATH ***
    dir_to_dif_filter4_raw = sc.textFile('s3://..../direct_to_diff_data_filter4.csv')# *** CHANGE TO CORRECT S3 PATH ***
    dir_to_dif_filter5_raw = sc.textFile('s3://..../direct_to_diff_data_filter5.csv')# *** CHANGE TO CORRECT S3 PATH ***
    dir_to_dif_filter6_raw = sc.textFile('s3://..../direct_to_diff_data_filter6.csv')# *** CHANGE TO CORRECT S3 PATH ***

    # Line : iDay (0), fDay (1), vari (2)
    variability_data_raw = sc.textFile('s3://..../variability_data.csv')# *** CHANGE TO CORRECT S3 PATH ***

    # Convert strings to floating point values
    days_all_rdd = days_all.map(parse_day_line)

    irr_filter1_rdd = irr_filter1_raw.map(irr_parse_line)
    irr_filter2_rdd = irr_filter2_raw.map(irr_parse_line)
    irr_filter3_rdd = irr_filter3_raw.map(irr_parse_line)
    irr_filter4_rdd = irr_filter4_raw.map(irr_parse_line)
    irr_filter5_rdd = irr_filter5_raw.map(irr_parse_line)
    irr_filter6_rdd = irr_filter6_raw.map(irr_parse_line)

    dir_to_dif_filter1_rdd = dir_to_dif_filter1_raw.map(dir_to_dif_irr_parse_line)
    dir_to_dif_filter2_rdd = dir_to_dif_filter2_raw.map(dir_to_dif_irr_parse_line)
    dir_to_dif_filter3_rdd = dir_to_dif_filter3_raw.map(dir_to_dif_irr_parse_line)
    dir_to_dif_filter4_rdd = dir_to_dif_filter4_raw.map(dir_to_dif_irr_parse_line)
    dir_to_dif_filter5_rdd = dir_to_dif_filter5_raw.map(dir_to_dif_irr_parse_line)
    dir_to_dif_filter6_rdd = dir_to_dif_filter6_raw.map(dir_to_dif_irr_parse_line)

    variability_rdd = variability_data_raw.map(var_parse_line)

# ------------------------------------------------------------------------------------------------
# DAILY FITTING
# ------------------------------------------------------------------------------------------------

# Fit the diffuse irradiance
# ------------------------------------------------------------------------------------------------

    # Line : iDayDif(0), fDayDif(1), vardif (2), dif(3)

    # Filter out cloudy days
    irr_filter1_rdd = irr_filter1_rdd.filter(irr_filter_sunny_skies)
    irr_filter2_rdd = irr_filter2_rdd.filter(irr_filter_sunny_skies)
    irr_filter3_rdd = irr_filter3_rdd.filter(irr_filter_sunny_skies)
    irr_filter4_rdd = irr_filter4_rdd.filter(irr_filter_sunny_skies)
    irr_filter5_rdd = irr_filter5_rdd.filter(irr_filter_sunny_skies)
    irr_filter6_rdd = irr_filter6_rdd.filter(irr_filter_sunny_skies)

    # Transform data to [(iDayDif1, [fDayDif, ...]),..,(iDayDifN, [fDayDif, ...])]
    irr_fitDayDif_filter1_rdd = irr_filter1_rdd.map(lambda line: (line[0], line[1])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDayDif_filter2_rdd = irr_filter2_rdd.map(lambda line: (line[0], line[1])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDayDif_filter3_rdd = irr_filter3_rdd.map(lambda line: (line[0], line[1])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDayDif_filter4_rdd = irr_filter4_rdd.map(lambda line: (line[0], line[1])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDayDif_filter5_rdd = irr_filter5_rdd.map(lambda line: (line[0], line[1])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDayDif_filter6_rdd = irr_filter6_rdd.map(lambda line: (line[0], line[1])).groupByKey().sortByKey()\
                                                .mapValues(list)

    # Transform data to [(iDayDif1, [dif, ...]),..,(iDayDifN, [dif, ...])]
    irr_fitDif_filter1_rdd = irr_filter1_rdd.map(lambda line: (line[0], line[3])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDif_filter2_rdd = irr_filter2_rdd.map(lambda line: (line[0], line[3])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDif_filter3_rdd = irr_filter3_rdd.map(lambda line: (line[0], line[3])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDif_filter4_rdd = irr_filter4_rdd.map(lambda line: (line[0], line[3])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDif_filter5_rdd = irr_filter5_rdd.map(lambda line: (line[0], line[3])).groupByKey().sortByKey()\
                                                .mapValues(list)
    irr_fitDif_filter6_rdd = irr_filter6_rdd.map(lambda line: (line[0], line[3])).groupByKey().sortByKey()\
                                                .mapValues(list)

    # Transform data to [(Day1, [[fDayDif1, fDayDif2,...,fDayDifN], [dif1, dif2,...,difN]]),...,
    #                    (DayN, [[fDayDif1, fDayDif2,...,fDayDifN], [dif1, dif2,...,difN]])]
    irr_union_filter1_rdd = sc.union([irr_fitDayDif_filter1_rdd, irr_fitDif_filter1_rdd]).groupByKey().sortByKey()\
                              .mapValues(list)
    irr_union_filter2_rdd = sc.union([irr_fitDayDif_filter2_rdd, irr_fitDif_filter2_rdd]).groupByKey().sortByKey() \
        .mapValues(list)
    irr_union_filter3_rdd = sc.union([irr_fitDayDif_filter3_rdd, irr_fitDif_filter3_rdd]).groupByKey().sortByKey() \
        .mapValues(list)
    irr_union_filter4_rdd = sc.union([irr_fitDayDif_filter4_rdd, irr_fitDif_filter4_rdd]).groupByKey().sortByKey() \
        .mapValues(list)
    irr_union_filter5_rdd = sc.union([irr_fitDayDif_filter5_rdd, irr_fitDif_filter5_rdd]).groupByKey().sortByKey() \
        .mapValues(list)
    irr_union_filter6_rdd = sc.union([irr_fitDayDif_filter6_rdd, irr_fitDif_filter6_rdd]).groupByKey().sortByKey() \
        .mapValues(list)

    # Filter out data that cannot be fit to a cosine curve
    irr_union_filter1_rdd = irr_union_filter1_rdd.filter(irr_filter_is_cosineable)
    irr_union_filter2_rdd = irr_union_filter2_rdd.filter(irr_filter_is_cosineable)
    irr_union_filter3_rdd = irr_union_filter3_rdd.filter(irr_filter_is_cosineable)
    irr_union_filter4_rdd = irr_union_filter4_rdd.filter(irr_filter_is_cosineable)
    irr_union_filter5_rdd = irr_union_filter5_rdd.filter(irr_filter_is_cosineable)
    irr_union_filter6_rdd = irr_union_filter6_rdd.filter(irr_filter_is_cosineable)

    # Collect the cosine parameters (p0 - p2) and the mean square error
    irr_cos_params_mse_filter1_rdd = irr_union_filter1_rdd.map(irr_get_cos_params_mse)
    irr_cos_params_mse_filter2_rdd = irr_union_filter2_rdd.map(irr_get_cos_params_mse)
    irr_cos_params_mse_filter3_rdd = irr_union_filter3_rdd.map(irr_get_cos_params_mse)
    irr_cos_params_mse_filter4_rdd = irr_union_filter4_rdd.map(irr_get_cos_params_mse)
    irr_cos_params_mse_filter5_rdd = irr_union_filter5_rdd.map(irr_get_cos_params_mse)
    irr_cos_params_mse_filter6_rdd = irr_union_filter6_rdd.map(irr_get_cos_params_mse)

    # Collect amplitude (p0)
    irr_p0_filter1_rdd = irr_cos_params_mse_filter1_rdd.map(lambda line: (line[0], line[1][0]))
    irr_p0_filter2_rdd = irr_cos_params_mse_filter2_rdd.map(lambda line: (line[0], line[1][0]))
    irr_p0_filter3_rdd = irr_cos_params_mse_filter3_rdd.map(lambda line: (line[0], line[1][0]))
    irr_p0_filter4_rdd = irr_cos_params_mse_filter4_rdd.map(lambda line: (line[0], line[1][0]))
    irr_p0_filter5_rdd = irr_cos_params_mse_filter5_rdd.map(lambda line: (line[0], line[1][0]))
    irr_p0_filter6_rdd = irr_cos_params_mse_filter6_rdd.map(lambda line: (line[0], line[1][0]))

    # Collect period (p1)
    irr_p1_filter1_rdd = irr_cos_params_mse_filter1_rdd.map(lambda line: (line[0], line[1][1]))
    irr_p1_filter2_rdd = irr_cos_params_mse_filter2_rdd.map(lambda line: (line[0], line[1][1]))
    irr_p1_filter3_rdd = irr_cos_params_mse_filter3_rdd.map(lambda line: (line[0], line[1][1]))
    irr_p1_filter4_rdd = irr_cos_params_mse_filter4_rdd.map(lambda line: (line[0], line[1][1]))
    irr_p1_filter5_rdd = irr_cos_params_mse_filter5_rdd.map(lambda line: (line[0], line[1][1]))
    irr_p1_filter6_rdd = irr_cos_params_mse_filter6_rdd.map(lambda line: (line[0], line[1][1]))

    # Collect phase offset (p2)
    irr_p2_filter1_rdd = irr_cos_params_mse_filter1_rdd.map(lambda line: (line[0], line[1][2]))
    irr_p2_filter2_rdd = irr_cos_params_mse_filter2_rdd.map(lambda line: (line[0], line[1][2]))
    irr_p2_filter3_rdd = irr_cos_params_mse_filter3_rdd.map(lambda line: (line[0], line[1][2]))
    irr_p2_filter4_rdd = irr_cos_params_mse_filter4_rdd.map(lambda line: (line[0], line[1][2]))
    irr_p2_filter5_rdd = irr_cos_params_mse_filter5_rdd.map(lambda line: (line[0], line[1][2]))
    irr_p2_filter6_rdd = irr_cos_params_mse_filter6_rdd.map(lambda line: (line[0], line[1][2]))

    # Collect mse
    irr_mse_filter1_rdd = irr_cos_params_mse_filter1_rdd.map(lambda line: (line[0], line[1][3]))
    irr_mse_filter2_rdd = irr_cos_params_mse_filter2_rdd.map(lambda line: (line[0], line[1][3]))
    irr_mse_filter3_rdd = irr_cos_params_mse_filter3_rdd.map(lambda line: (line[0], line[1][3]))
    irr_mse_filter4_rdd = irr_cos_params_mse_filter4_rdd.map(lambda line: (line[0], line[1][3]))
    irr_mse_filter5_rdd = irr_cos_params_mse_filter5_rdd.map(lambda line: (line[0], line[1][3]))
    irr_mse_filter6_rdd = irr_cos_params_mse_filter6_rdd.map(lambda line: (line[0], line[1][3]))

# Fit the direct-to-diffuse irradiance
# ------------------------------------------------------------------------------------------------

    # Line : iDayDdrf (0), fDayDdrf (1), ddrf (2), varddrf (3)

    # Filter out turbulent skies
    dir_to_dif_filter1_rdd = dir_to_dif_filter1_rdd.filter(dir_to_dif_stable_skies)
    dir_to_dif_filter2_rdd = dir_to_dif_filter2_rdd.filter(dir_to_dif_stable_skies)
    dir_to_dif_filter3_rdd = dir_to_dif_filter3_rdd.filter(dir_to_dif_stable_skies)
    dir_to_dif_filter4_rdd = dir_to_dif_filter4_rdd.filter(dir_to_dif_stable_skies)
    dir_to_dif_filter5_rdd = dir_to_dif_filter5_rdd.filter(dir_to_dif_stable_skies)
    dir_to_dif_filter6_rdd = dir_to_dif_filter6_rdd.filter(dir_to_dif_stable_skies)

    # Transform data to [(iDayDdrf, (ddrf, ...)), ...] and filter out days not containing enough
    # measurements
    dir_to_dif_filter1_rdd = dir_to_dif_filter1_rdd.map(dir_to_dif_get_day_ddrf_pair).groupByKey().mapValues(list)\
                                                       .filter(dir_to_dif_min_meas_thresh)

    dir_to_dif_filter2_rdd = dir_to_dif_filter2_rdd.map(dir_to_dif_get_day_ddrf_pair).groupByKey().mapValues(list)\
                                                       .filter(dir_to_dif_min_meas_thresh)

    dir_to_dif_filter3_rdd = dir_to_dif_filter3_rdd.map(dir_to_dif_get_day_ddrf_pair).groupByKey().mapValues(list)\
                                                       .filter(dir_to_dif_min_meas_thresh)

    dir_to_dif_filter4_rdd = dir_to_dif_filter4_rdd.map(dir_to_dif_get_day_ddrf_pair).groupByKey().mapValues(list)\
                                                       .filter(dir_to_dif_min_meas_thresh)

    dir_to_dif_filter5_rdd = dir_to_dif_filter5_rdd.map(dir_to_dif_get_day_ddrf_pair).groupByKey().mapValues(list)\
                                                       .filter(dir_to_dif_min_meas_thresh)

    dir_to_dif_filter6_rdd = dir_to_dif_filter6_rdd.map(dir_to_dif_get_day_ddrf_pair).groupByKey().mapValues(list)\
                                                       .filter(dir_to_dif_min_meas_thresh)

    # Collect average of ddrf for day
    dir_to_dif_avg_filter1_rdd = dir_to_dif_filter1_rdd.map(dir_to_dif_get_avg)
    dir_to_dif_avg_filter2_rdd = dir_to_dif_filter2_rdd.map(dir_to_dif_get_avg)
    dir_to_dif_avg_filter3_rdd = dir_to_dif_filter3_rdd.map(dir_to_dif_get_avg)
    dir_to_dif_avg_filter4_rdd = dir_to_dif_filter4_rdd.map(dir_to_dif_get_avg)
    dir_to_dif_avg_filter5_rdd = dir_to_dif_filter5_rdd.map(dir_to_dif_get_avg)
    dir_to_dif_avg_filter6_rdd = dir_to_dif_filter6_rdd.map(dir_to_dif_get_avg)

    # Collect standard deviation of ddrf
    dToDIrr_std_dev_filter1_rdd = dir_to_dif_filter1_rdd.map(dir_to_dif_get_std_dev)
    dToDIrr_std_dev_filter2_rdd = dir_to_dif_filter2_rdd.map(dir_to_dif_get_std_dev)
    dToDIrr_std_dev_filter3_rdd = dir_to_dif_filter3_rdd.map(dir_to_dif_get_std_dev)
    dToDIrr_std_dev_filter4_rdd = dir_to_dif_filter4_rdd.map(dir_to_dif_get_std_dev)
    dToDIrr_std_dev_filter5_rdd = dir_to_dif_filter5_rdd.map(dir_to_dif_get_std_dev)
    dToDIrr_std_dev_filter6_rdd = dir_to_dif_filter6_rdd.map(dir_to_dif_get_std_dev)

# Fit the variability
# ------------------------------------------------------------------------------------------------

    # Line : iDay (0), fDay (1), vari (2)

    # Filter out days containing cloudy skies
    var_filter1_rdd = variability_rdd.filter(var_filter_clear_skies)

    # Transform variablity data to use iDay as key and
    # vari as value. Group by key.
    # [(iDay, [vari1, vari2,...,varin]), ...]
    var_filter1_rdd = var_filter1_rdd.map(var_get_day_vari_pair).groupByKey().mapValues(list)

    # Filter out days not containing enough measurements
    var_filter1_rdd = var_filter1_rdd.filter(var_filter_min_meas_thresh)

    # Collect average of vari for the day
    var_day_avg_rdd = var_filter1_rdd.map(var_get_avg)

    # Collect standard deviation of vari for the day
    var_day_std_dev_rdd = var_filter1_rdd.map(var_get_std_dev)

# Collect feature table
# ------------------------------------------------------------------------------------------------
    # Each line of feature table represents a given day and data passing cuts for all filters (1 - 6)

    feature_table_rdd = sc.union([
        # Irradiance
        irr_p0_filter1_rdd, irr_p1_filter1_rdd, irr_p2_filter1_rdd, irr_mse_filter1_rdd,
        irr_p0_filter2_rdd, irr_p1_filter2_rdd, irr_p2_filter2_rdd, irr_mse_filter2_rdd,
        irr_p0_filter3_rdd, irr_p1_filter3_rdd, irr_p2_filter3_rdd, irr_mse_filter3_rdd,
        irr_p0_filter4_rdd, irr_p1_filter4_rdd, irr_p2_filter4_rdd, irr_mse_filter4_rdd,
        irr_p0_filter5_rdd, irr_p1_filter5_rdd, irr_p2_filter5_rdd, irr_mse_filter5_rdd,
        irr_p0_filter6_rdd, irr_p1_filter6_rdd, irr_p2_filter6_rdd, irr_mse_filter6_rdd,
        # Direct-to-Diffuse Irradiance
        dir_to_dif_avg_filter1_rdd, dToDIrr_std_dev_filter1_rdd,
        dir_to_dif_avg_filter2_rdd, dToDIrr_std_dev_filter2_rdd,
        dir_to_dif_avg_filter3_rdd, dToDIrr_std_dev_filter3_rdd,
        dir_to_dif_avg_filter4_rdd, dToDIrr_std_dev_filter4_rdd,
        dir_to_dif_avg_filter5_rdd, dToDIrr_std_dev_filter5_rdd,
        dir_to_dif_avg_filter6_rdd, dToDIrr_std_dev_filter6_rdd,
        # Variability
        var_day_avg_rdd, var_day_std_dev_rdd
    ])

    # Create rows of feature table by grouping by day in ascending order and
    # append day as first element
    feature_table_rdd = feature_table_rdd.groupByKey().sortByKey().mapValues(list).map(day_append_beginning)

    # Filter out rows not containing all features
    feature_table_rdd = feature_table_rdd.filter(lambda line: len(line) == FEAT_TABLE_TOTAL_COLS)

# ------------------------------------------------------------------------------------------------
# SEASONAL FITTING
# ------------------------------------------------------------------------------------------------

# Create DataFrames
# ------------------------------------------------------------------------------------------------

    input_data = feature_table_rdd.map(lambda d: (d[0], d[1], DenseVector(d[2:])))

    df = sqlContext.createDataFrame(input_data, [DAY_COL_TITLE, LABEL_COL_TITLE, FEATURE_COL_TITLE])

# Assemble the training and test sets
# ------------------------------------------------------------------------------------------------

    train_data_range = df.where((df.day >= TRAIN_RANGE_DAYS[0]) & (df.day <= TRAIN_RANGE_DAYS[1]))

    test_data_range = df.where((df.day < TRAIN_RANGE_DAYS[0]) | (df.day > TRAIN_RANGE_DAYS[1]))

    # Save Day for plotting
    test_data_Day = test_data_range.select(DAY_COL_TITLE)

    # Collect label and feature columns for training and testing model
    train_data = train_data_range.select(LABEL_COL_TITLE, FEATURE_COL_TITLE)
    test_data = test_data_range.select(LABEL_COL_TITLE, FEATURE_COL_TITLE)

# Train the model
# ------------------------------------------------------------------------------------------------

    # Initialize regression
    lr = LinearRegression(labelCol=LABEL_COL_TITLE, maxIter=10)

    # Fit the data to the model
    linear_model = lr.fit(train_data)

# Test the model
# ------------------------------------------------------------------------------------------------

    # Generate predictions
    predicted = linear_model.transform(test_data)

    # Extract the predictions and the "known" correct labels
    predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
    labels = predicted.select(LABEL_COL_TITLE).rdd.map(lambda x: x[0])

    # Zip predictions and labels into a list
    predictionAndLabel = predictions.zip(labels)

    # Compute residual sum of squares
    rsos = predictionAndLabel.map(lambda p_l: np.mean((p_l[0] - p_l[1]) ** 2))

    # The smaller the RMSE value, the closer predicted and observed values are
    root_mean_squared_error = linear_model.summary.rootMeanSquaredError
    print('Root Mean Squared Error = ', root_mean_squared_error)

    # The higher the R-squared, the better the model fits data
    r_squared = linear_model.summary.r2
    print('R Squared = ', r_squared)

# Save results to S3
# ------------------------------------------------------------------------------------------------

    # Collect test days
    test_data_days = test_data_Day.rdd.map(lambda x : x[0])

    # Save to S3 bucket
    test_data_days.saveAsTextFile('s3://..../mfrsr_app_days_output')# *** CHANGE TO CORRECT S3 PATH ***
    rsos.saveAsTextFile('s3://..../mfrsr_app_rsos_output')# *** CHANGE TO CORRECT S3 PATH ***

# Terminate Spark session
# ------------------------------------------------------------------------------------------------
    sc.stop()