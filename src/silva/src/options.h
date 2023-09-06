/**
 * Defines program options.
 *
 * Reads and sets program options.
 *
 * @file options.h
 * @author Marco Zanella <marco.zanella.1991@gmail.com>
 */
#ifndef OPTIONS_H
#define OPTIONS_H

#include <stdio.h>

#include "forest.h"
#include "perturbation.h"
#include "tier.h"
#include "abstract_domains/abstract_domain.h"


/** Type of program options. */
typedef struct options Options;


/** Structure of program options. */
struct options {
    char *classifier_path;             /**< Path to classifier file. */
    char *dataset_path;                /**< Path to dataset file. */
    unsigned int max_print_length;     /**< Maximum number of characters to show
                                            for classifier and dataset paths. */
    int index_instance_to_verify;      /**< Index of the instance of the test set to verify, -1 if all */
    unsigned long int memory_soft_limit; /**< Limit on the memory usage in bytes, 0 if no limit */ 
    ForestVotingScheme voting_scheme;  /**< Forest voting scheme. */
    AbstractDomain abstract_domain;    /**< Abstract domain to use for analysis. */
    Perturbation perturbation;         /**< Type of perturbation. */
    Tier tier;                         /**< Tier list of features. */
    unsigned int sample_timeout;       /**< Maximum allowed execution time for
                                            one sample analysis (seconds). */
    unsigned int seed;                 /**< Seed to use for random number
                                            generator. */
};



void options_delete(Options *options);

/**
 * Reads command-line options.
 *
 * @param[out] options Program options
 * @param[in] argc ARGument Counter
 * @param[in] argv ARGument Vector
 */
void options_read(Options *options, const int argc, const char *argv[]);


/**
 * Prints a help message.
 *
 * @param[in] argc ARGument Counter
 * @param[in] argv ARGument Vector
 */
void display_help(const int argc, const char *argv[]);


/**
 * Prints program options.
 *
 * @param[in] options Program options
 * @param[in,out] stream Stream
 */
void options_print(const Options options, FILE *stream);

#endif

