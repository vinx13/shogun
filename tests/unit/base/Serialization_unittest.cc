/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Thoralf Klein, Evgeniy Andreev, Soeren Sonnenburg, 
 *          Heiko Strathmann
 */

#include <gtest/gtest.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

TEST(Serialization,multiclass_labels)
{
	index_t n=10;
	index_t n_class=3;

	CMulticlassLabels* labels=new CMulticlassLabels();

	SGVector<float64_t> lab(n);
	for (index_t i=0; i<n; ++i)
		lab[i]=i%n_class;

	labels->set_labels(lab);

	labels->allocate_confidences_for(n_class);
	SGVector<float64_t> conf(n_class);
	for (index_t i=0; i<n_class; ++i)
		conf[i]=CMath::randn_double();

	for (index_t i=0; i<n; ++i)
		labels->set_multiclass_confidences(i, conf);

	/* create serialized copy */
	const char* filename="multiclass_labels.txt";
	CSerializableAsciiFile* file=new CSerializableAsciiFile(filename, 'w');
	labels->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename, 'r');
	CMulticlassLabels* labels_loaded=new CMulticlassLabels();
	labels_loaded->load_serializable(file);
	file->close();
	SG_UNREF(file);

	/* compare */
	for (index_t i=0; i<n; ++i)
		ASSERT(labels_loaded->get_labels()[i]==labels->get_labels()[i]);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n_class; ++j)
		{
			//float64_t a=labels->get_multiclass_confidences(i)[j];
			//float64_t b=labels_loaded->get_multiclass_confidences(i)[j];
			// Add one multiclass serialization works
			//float64_t diff=CMath::abs(a-b);
			//EXPECT_LE(diff, 10E-15);
		}
	}

	SG_UNREF(labels_loaded);
	SG_UNREF(labels);
}

#ifdef HAVE_LAPACK
TEST(Serialization, liblinear)
{
	index_t num_samples = 50;
	CMath::init_random(13);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CLibLinear* liblin = new CLibLinear(1.0, train_feats, ground_truth);
	liblin->set_epsilon(1e-5);
	liblin->fit(train_feats, ground_truth);;

	CBinaryLabels* pred = liblin->predict(test_feats)->as<CBinaryLabels>();
	for (int i = 0; i < num_samples; ++i)
		EXPECT_EQ(ground_truth->get_int_label(i), pred->get_int_label(i));
	SG_UNREF(pred);

	/* save liblin */
	const char* filename="trained_liblin.txt";
	CSerializableAsciiFile* file=new CSerializableAsciiFile(filename, 'w');
	liblin->save_serializable(file);
	file->close();
	SG_UNREF(file);

	/* load liblin */
	file=new CSerializableAsciiFile(filename, 'r');
	CLibLinear* liblin_loaded=new CLibLinear();
	liblin_loaded->load_serializable(file);
	file->close();
	SG_UNREF(file);

	/* classify with the deserialized model */
	pred = liblin_loaded->apply(test_feats)->as<CBinaryLabels>();
	for (int i = 0; i < num_samples; ++i)
		EXPECT_EQ(ground_truth->get_int_label(i), pred->get_int_label(i));

	SG_UNREF(liblin_loaded);
	SG_UNREF(liblin);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}
#endif
