KUBIG Contest

<Data>
The dataset consists of data collected from heavy Scania trucks in everyday usage. The system
in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in
various functions in a truck, such as braking and gear changes. The dataset’s positive class
consists of component failures for a specific component of the APS system. The negative class
consists of trucks with failures for components not related to the APS. The data consists of a
subset of all available data, selected by experts.

Dataset Details :
The training set contains 60000 examples in total in which
59000 belong to the negative class and 1000 positive class.
The test set contains 16000 examples.
•
Number of Attributes : 171
•
Attribute Information: The attribute names of the data have been anonymized for proprietary reasons. It consists of both
single numerical counters and histograms consisting of bins with different conditions. Typically the histograms have open
ended conditions at each end. For example if we measuring the ambient temperature ‘T’ then the histogram could be
defined with 4 bins where:
•
bin 1 collect values for temperature T < 20
•
bin 2 collect values for temperature T >= 20 and T < 0
•
bin 3 collect values for temperature T >= 0 and T < 20
•
bin 4 collect values for temperature T > 20
•
The attributes are as follows: class, then anonymized operational data. The operational data have an identifier and a bin id,
like ‘ Identifier_Bin ’. In total there are 171 attributes, of which 7 are histogram variables. Missing values are denoted by na
