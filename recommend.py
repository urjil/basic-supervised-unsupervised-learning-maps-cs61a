"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    # BEGIN Question 3
    "*** YOUR CODE HERE ***"
    #dist_dict={x: distance(location,x) for x in centroids}
    #return key_of_min_value(dist_dict)
    #distance(location, x) for x in centroids
    return min(centroids,key=lambda centroid: distance(location,centroid))

    # lowest=distance(location,centroids[0])
    # pair=centroids[0]


    # END Question 3


def group_by_first(pairs):
    """Return a list of lists that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)  # Values from pairs that start with 1, 3, and 2 respectively
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # BEGIN Question 4
    "*** YOUR CODE HERE ***"
    # END Question 4
    #for x in restaurants:
    #    lowest=100000
    #    for y in centroids:
    #        #find the min and assign it to a list
  
    rest_locs=[restaurant_location(x) for x in restaurants]
    storage=[]

    #for x in rest_locs:
    #    storage= storage+[find_closest(x,centroids)]
    #
    [storage.append(find_closest(x,centroids)) for x in rest_locs]


    rest_names=zip(storage,restaurants)


    return group_by_first(rest_names)


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    # BEGIN Question 5
    "*** YOUR CODE HERE ***"
    #is cluster name of restaurant or location?
    

    lat_lon=[]
    for x in cluster:
        lat_lon.append(restaurant_location(x))
    sum_lat=0
    sum_lon=0
    for x in lat_lon:
        sum_lat=x[0]+sum_lat
        sum_lon=x[1]+sum_lon
    average_lat, average_lon= float((sum_lat/len(lat_lon))), float((sum_lon/len(lat_lon)))

    return [average_lat,average_lon]



    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0

    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        # BEGIN Question 6
        "*** YOUR CODE HERE ***"
        #centroids= [group_by_centroid(restaurants,x) for x in centroids]
        
        rest_names= group_by_centroid(restaurants,centroids)
        centroids= [find_centroid(x) for x in rest_names]
        #centroids=find_centroid(centroids)
        #group_by_centroid(centroids)
        # END Question 6
        n += 1
    return centroids


################################
# Phase 3: Supervised Learning #
################################


def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    xs = [feature_fn(r) for r in restaurants]
    ys = [user_rating(user, restaurant_name(r)) for r in restaurants]

    # BEGIN Question 7
    "*** YOUR CODE HERE ***"
    #S_xx, S_yy, and S_xy
    #Sxx = Σi (xi - mean(x))2

    mean_x=mean(xs)
    mean_y=mean(ys)
     
    diff_x=[x-mean_x for x in xs]
    diff_y=[y-mean_y for y in ys]

    squared_diff_x=[x*x for x in diff_x]
    squared_diff_y=[y*y for y in diff_y]

    S_xx=sum(squared_diff_x)
    S_yy=sum(squared_diff_y)
    S_xy=sum([x*y for x, y in zip(diff_x,diff_y)])   #diff_x, diff_y])
    #zip(diff_x,diff_Y)

    b= S_xy / S_xx
    a = mean_y - b * mean_x
    r_squared = S_xy**2 / (S_xx * S_yy)

    #b = Sxy / Sxx



    #S_xx=[     for x in diff_x]
    # END Question 7

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 8
    "*** YOUR CODE HERE ***"
    # [x[reviewed] for x in feature_fns]  #runs function in feature_fns on the list of all restaurants
    # [feature_fns[x] for x in reviewed]   # runs one restaurant after the other on a set of functions.
    # max(___, key= r2 value ) 


    test=[]
    #run every predictor on a restuarant, then run this on all the restaurants s
    for x in feature_fns:
        #test=test+find_predictor(user, reviewed,x) # we can also add predictor to the list
        test.append(find_predictor(user, reviewed,x))

    
    return max(test, key=lambda x:x[1])[0]

    #return max(feature_fns, key= lambda x: find_predictor(user, reviewed,x)[1])[0]  
    # END Question 8


def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 9
    "*** YOUR CODE HERE ***"
    #[restaurant_name(x) for x in ALL_RESTAURANTS]

    # if restaurant exists in reviewed, then set value to user_rating(user, restaurant_name)
    #[x: prediction for x in restaurants]
    #{x: }
    rest_names=[restaurant_name(x) for x in restaurants]

    ratings=[]
    for x in restaurants:
        if x in reviewed:
            ratings.append(user_rating(user,restaurant_name(x)))
        else:
            ratings.append(predictor(x))
    
    return {x:y for x,y in zip(rest_names,ratings)}




    # END Question 9


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    "*** YOUR CODE HERE ***"
    search_result=[]
    for x in restaurants:
        if query in restaurant_categories(x):
            search_result.append(x)

    return search_result


    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)