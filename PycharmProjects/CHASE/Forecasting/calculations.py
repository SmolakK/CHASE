from data_reader import geo_to_df, water_to_df
from preprocessing import regular_week, embed, best_decay_offset
from water import calculate_best

calculate_best('RF',10,'2017-10-01','2017-12-01','2017-12-01','2017-12-15')
