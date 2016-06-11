import main
import multi_training
import model
import cPickle as pickle

pcs = multi_training.loadPieces("music")

print "building model"
m = model.Model([300,300],[100,50], dropout=0.5)

m.learned_config=pickle.load(open( "output/final_learned_config.p", "rb" ))

main.gen_adaptive(m,pcs,10,name="composition")