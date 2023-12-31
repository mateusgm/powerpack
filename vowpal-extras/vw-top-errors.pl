e#!/usr/bin/perl -w
# vim: ts=4 sw=4 nosmarttab
#
# Goal: improve learning (smaller generalization error)
#
# How:  Inspect the learning process in real-time. Single-out
#       the elements leading to its biggest jumps in loss
#       (deviations from convergence) and drill-down to uncover
#       the features that underlie these biggest deviations.
#
# The process has two steps:
#   1) Train with --progress 1 to capture the examples that cause
#      the biggest divergences (increase in progressive loss) in
#      the learning process.
#   2) Audit the examples from step 1) and drill down to their
#      top weighted features (the features that presumably are
#      throwing the learning process off course: i.e. cause 'un-learning').
#
# First step:
#   input:    VW training data and args   (to generate stderr with -P 1)
#   output:   N biggest error pairs (example_number   delta_error_since_last)
#
# Second step:
#   input:    The same training set and args as first step
#             plus the output from the 1st step (N biggest errors)
#   output:   A dump of the biggest-error examples each with its
#             top-N highest weight features.
#
# Caveats:
#   Running vw-top-errors only points-out the top suspects.
#   top suspects are not necessarily "guilty".
#   For example: if the constant feature is large and has a large
#   relative weight, it may be singled out as a strong-feature in
#   examples that are deviant.  It is entirely possible that it is
#   also a strong feature in other/most examples.
#
#   So use and enjoy this utility, but be aware of its limitations.
#
# -- Copyright (c) ariel faigon, 2014
# This software is released under the same licence as vowpal-wabbit.
# itself.  See the file LICENSE.
#

use Getopt::Std;
use vars qw($opt_v $opt_V $opt_s $opt_a);

my %ExampleDelta = ();
my $TopN;
my $TopWeights;
my %ExampleNos;
my @ExampleNos;

my $DefaultTopN = 10;
my $DefaultTopWeights = 5;

my $VWCmd;

sub v {
    return unless $opt_v;
    if (@_ == 1) {
        print STDERR @_;
    } else {
        printf STDERR @_;
    }
}

sub V {
    return unless $opt_V;
    if (@_ == 1) {
        print STDERR @_;
    } else {
        printf STDERR @_;
    }
}

sub usage(@) {
    print STDERR @_, "\n" if @_;
    die "Usage: $0 [options] [args] vw <vw-training-arguments>...

    Will run 2 passes:
    1) Learning with --progress 1: catch the biggest errors
    2) Learning with --audit: intercept the biggest errors
       and print their highest weights.

    $0 options (must precede the 'vw' and its args):
        -s      Use sum-of-weights instead of mean-of-weights
                in 'top adverse features' summary
        -a      Use absolute rather than (the default) relative error
                (since-last / mean loss) for top N deviant examples
        -v      verbose

    $0 args (must precede the 'vw' and its args):
        <TopN>          Integer: how many (top error) examples to print
                        (Default: $DefaultTopN)
        <TopWeights>    Integer: how many top weights to print for each
                        (top error) example
                        (Default: $DefaultTopWeights)
";
}

sub get_args {
    $0 =~ s{.*/}{};

    # Separate our args, from vw and its args...
    my (@our_args, @vw_args);
    my $saw_vw = 0;
    foreach my $arg (@ARGV) {
        if ($saw_vw) {
            push(@vw_args, $arg);
            next;
        } elsif ($arg =~ /^\S*vw$/) {
            $saw_vw = 1;
            push(@vw_args, $arg);
            next;
        }
        if ($arg =~ /^\d+$/) {
            if (! defined $TopN) {
                $TopN = $arg;
            } elsif (! defined $TopWeights) {
                $TopWeights = $arg;
            } else {
                printf STDERR "Too many integer args: ignoring $arg\n"
            }
        }
        push(@our_args, $arg);
    }

    $TopN = $DefaultTopN unless (defined $TopN);
    $TopWeights = $DefaultTopWeights unless (defined $TopWeights);

    unless (@vw_args >= 2) {
        usage("Expecting a 'vw' argument followed by training-args");
    }
    $VWCmd = "@vw_args";

    @ARGV = @our_args;
    getopts('vsa');

    # Make wide-char output safe from warnings
    binmode STDOUT, ":utf8";
}

#
# collect_errors
#   Reads vw progress output and collects delta-loss magnitudes
#   in the hash %ExampleDelta.
#
#   We're looking for positive deltas, i.e. examples where the
#   loss went up instead of down.
#
sub collect_errors($) {
    my $vw_stderr = shift;

    my $good_lines = 0;
    my ($prev_avgloss, $prev_sincelast);
    my $stderr = '';
    while (<$vw_stderr>) {
        unless (/^([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s/) {
            # Not a progress line, may be needed if 'vw' crashes etc.
            $stderr .= "\t$_";
            next;
        }
        $good_lines++;
        my ($avgloss, $sincelast, $example_no) = ($1, $2, $3);
        if (defined $prev_sincelast) {
            my $delta_since_last = $sincelast - $prev_sincelast;
            if ($opt_a) {
                # absolute error (rare need)
                $ExampleDelta{$example_no} = $delta_since_last;
            } else {
                # relative error (default)
                my $relative_error = $delta_since_last / $avgloss;
                $ExampleDelta{$example_no} = $relative_error;
            }
        }

        ($prev_avgloss, $prev_sincelast) = ($avgloss, $sincelast);
    }
    wait;
    if ($?) {
        printf STDERR "%s: vw exited w/ status %d\nvw stderr:\n%s\n",
                        $0, $? >> 8, $stderr;
        exit 1;
    }
    unless ($good_lines > 0) {
        printf STDERR "%s: no progress lines from vw training\n%s\n",
                        $0, $stderr;
        exit 1;
    }
}

#
# by_abs_weight_desc($a, $b)
#   compare function to sort features by abs(weight) largest first
#
sub by_abs_weight_desc {
    my $weight1 = (split(':', $a))[-1];
    my $weight2 = (split(':', $b))[-1];
    $weight1 =~ s/\@.*$//;
    $weight2 =~ s/\@.*$//;

    abs($weight2) <=> abs($weight1);
}

#
# audit_top_weights($vwh, $top_weights, @example_nos);
#   $vwh            file handle of vw audit stream
#   $top_weights    number of weights to print per example
#   @example_nos    example numbers to cover
#
sub audit_top_weights($$@) {
    my ($vw_stderr, $top_weights, @example_nos) = @_;

    unless (@example_nos) {
        die "$0: audit_top_weights: no wanted examples given\n";
    }

    # Make sure we have example numbers in order from low to high
    @example_nos = sort { $a <=> $b } @example_nos;

    my %feature_weights = ();
    my %feature_examples = ();

    printf "=== Top-weighting underlying %d features" .
            " @ diverging examples:\n", $top_weights;

    # -- Scan the vw audit output
    my $example_count = 0;
    while (<$vw_stderr>) {
        last unless (@example_nos);

        # No need to skip regular progress lines because
        # we rn with --quiet
        # next if (/^[0-9.]+\s+[0-9.]+\s+\d+\s+[0-9]+\.[0-9]\s+/);

        # Look for an audit-preceding line
        next unless (/^-?[0-9.]+\s/);

        # Found an example:
        $example_count++;

        my $wanted_example = $example_nos[0];

        if ($example_count == $wanted_example) {
            # Found the next wanted one
            shift @example_nos;

            # -- print the matching label/tag etc. header
            printf "Example #%d: %s", $wanted_example, $_;

            # Next line has the features and weights
            my $audit_line = <$vw_stderr>;
            chomp $audit_line;
            $audit_line =~ s/^\s+//;

            my @features = split(/\s+/, $audit_line);
            my @sorted_features = sort by_abs_weight_desc @features;

            printf "\t%s\t%-8s\t%s\n", 'Feature', 'Weight', 'Full-audit-data';
            for (my $i = 0; $i < $TopWeights; $i++) {
                my $feature = $sorted_features[$i];
                next unless (defined $feature);

                my ($name, $hash, $value, $weight) = split(':', $feature);
                unless (defined $weight) {
                    # multiple passes (using cache) don't have the
                    # name but only the hashed value of it so
                    # fields get shifted by one...
                    printf STDERR "Undefined weight in audit data: feature=%s\n", $feature;
                    $weight = $value;
                    $value = $hash;
                    $hash = $name;
                    $name = "[$hash]";
                }
                $weight =~ s/\@.*$//;
                printf "\t%s\t%7.6f\t%s\n", $name, $weight, $feature;

                $feature_weights{$name} += abs($weight);

                unless (exists $feature_examples{$name}) {
                    $feature_examples{$name} = {};
                }
                $feature_examples{$name}->{$wanted_example} = $weight;
            }
        }
    }
    print "\n";
    printf "=== Top adverse features (in descending %s weight)\n",
                            ($opt_s ? 'cummulative' : 'mean');
    printf "%-20s\t%-10s  %s\n", 'Name',
                                 ($opt_s ? 'Sum-Weight' : 'Mean-Weight'),
                                 'Examples';

    unless ($opt_s) {   # Switch from sum of-weights to mean
        foreach my $name (keys %feature_weights) {
            my @examples = keys %{$feature_examples{$name}};
            my $example_count = scalar @examples;
            $feature_weights{$name} /= $example_count;
        }
    }

    foreach my $name (sort
                        {$feature_weights{$b} <=> $feature_weights{$a}}
                            keys %feature_weights) {
        
        my @example_list = sort {
                                $feature_examples{$name}->{$b}
                                    <=>
                                $feature_examples{$name}->{$a}
                            }
                            keys %{$feature_examples{$name}};

        printf "%-20s\t%-10.6f  @example_list\n",
                $name, $feature_weights{$name};
    }
}

#
# sort delta loss numerically, descending
#
sub by_delta() {
    $ExampleDelta{$b} <=> $ExampleDelta{$a};
}

#
# biggest_error_examples(N)
#   list of top N loss example_numbers
#
sub biggest_error_examples($) {
    my $howmany = shift;

    my @sorted_examples = sort by_delta keys %ExampleDelta;

    @sorted_examples[0 .. $howmany-1];
}

#
# print_errors(@examples)
#   Print the top N loss example numbers and their absolute or
#   relative loss.
#
sub print_errors(@) {
    printf "=== Top-%d (highest delta loss) diverging examples:\n", scalar(@_);
    printf "Example\t%s-Loss\n", ($opt_a ? 'Absolute' : 'Relative');
    foreach my $example (@_) {
        printf "%d\t%g\n", $example, $ExampleDelta{$example};
    }
    print "\n";
}

#
# first_pass(N)
#   First training pass to capture progressive loss numbers.
#
sub first_pass($) {
    my $top_n = shift;

    my $vw_cmd = $VWCmd;
    $vw_cmd =~ s/(?:--progress|-P)\s*[0-9.]+\b//;
    $vw_cmd =~ s/(?:--quiet\b)//;
    $vw_cmd .= " --progress 1";

    v("+----- 1st pass: find top %d errors\n", $top_n);
    v("| 1st pass: \$vw_cmd: %s\n", $vw_cmd);
    open(my $vwh, "$vw_cmd 2>&1 |");
    collect_errors($vwh);
    close $vwh;
    my @top_error_examples = biggest_error_examples($top_n);
    print_errors(@top_error_examples);
    v("+--- 1st pass: done!\n\n");

    @top_error_examples;
}

#
# second_pass($top_weights, @example_nos)
#   2nd training pass with --audit to capture individual features
#   causing the biggest loss jumps.
#
sub second_pass($@) {
    my ($top_weights, @example_nos) = @_;

    v("+----- 2nd pass: --audit to find top bad features\n");
    v("| 2nd pass: \$top_weights: %d\n", $top_weights);
    v("| 2nd pass: \@example_nos: @example_nos\n");

    my $vw_cmd = $VWCmd;
    $vw_cmd =~ s/(?:--progress|-P)\s*[0-9.]+\b//;
    $vw_cmd =~ s/(?:--quiet\b)//;
    $vw_cmd =~ s/(?:--audit|-a\b)//;
    $vw_cmd .= " --quiet --audit";

    v("| 2nd pass: \$vw_cmd: %s\n", $vw_cmd);
    open(my $vwh, "$vw_cmd 2>&1 |");
    audit_top_weights($vwh, $top_weights, @example_nos);
    close $vwh;
    v("+--- 2nd pass: done!\n");
}

# -- main
get_args;
my @top_errors = first_pass($TopN);
second_pass($TopWeights, @top_errors);


