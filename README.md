# maps
This is a project that I implemented for my CS61A class.

In this project, I created a visualization of restaurant ratings using basic machine learning (supervised and unsupervised learning) and the Yelp academic dataset. 
In this visualization, Berkeley is segmented into regions, where each region is shaded by the predicted rating of the closest restaurant (yellow is 5 stars, blue is 1 star).
Specifically, the visualization constructed is a Voronoi diagram.

In the map each dot represents a restaurant. The color of the dot is determined by the restaurant's location. For example, downtown restaurants are colored green.
The user that generated this map has a strong preference for Southside restaurants, and so the southern regions are colored yellow.


DISCLAIMER: 
The project uses several files, but I have only made changess to utils.py, abstractions.py, and recommend.py, the rest of the files contain skeleton code written by the course staff.



