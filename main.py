import os
import argparse
import torch
import numpy as np
from model import Net
from dataset import InputDataset
from torch.utils.data import DataLoader
from trainer import Trainer

def main(args):
    dataset = InputDataset(args.root, args.focal_length_X, args.focal_const, args.num_points)
    camera_matrix = np.array([args.focal_length_X, 0, args.principal_point_X, 0, args.focal_length_Y, args.principal_point_Y, 0, 0, 1])
    camera_matrix = camera_matrix.reshape(3, -1)
    train_data, valid_data, test_data = torch.utils.data.random_split(dataset=dataset, lengths=[(1-args.test_size)**2*len(dataset), (1-args.test_size)*args.test_size*len(dataset), args.test_size*len(dataset)])
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    
    model = Net(args.num_points, camera_matrix)
    trainer = Trainer(args, model)

    print("-------Train Start------")
    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train_loss = trainer.train(train_loader) 
        train_succ = trainer.evaluate(train_loader)
        valid_loss, valid_succ = trainer.evaluate(valid_loader)
        valid_loss, valid_succ = 0.0, 0.0
        print("Epoch[{}/{}], Train Loss : {:.4f}, Train Succ : {:.2f}%, Valid Loss : {:.4f}, Valid Succ : {:.2f}%".format(epoch + 1, args.num_epochs, train_loss, train_succ, valid_loss, valid_succ))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "epoch_{}.pth".format(epoch + 1))

    print("-------Train Ended------")

    model.load_state_dict(torch.load('./epoch_{}.pth'.format(best_epoch)))
    test_loss, test_succ = trainer.evaluate(test_loader)
    print("\n[Using Epoch {}'s model, evaluate on Test set]".format(best_epoch))
    print("Test Success rate:{:.2f}% (Loss:{:.4f})".format(test_succ, test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=20, type=int, help="The num of epochs for training")
    parser.add_argument("--num_points", default=3, type=int, help="The num of keypoints in an image")
    parser.add_argument("--num_imgs", default=2, type=int, help="The num of images")
    parser.add_argument("--focal_length_X", default=1, type=int, help="focal length of the camera(x)")
    parser.add_argument("--focal_length_Y", default=1, type=int, help="focal length of the camera(y)")
    parser.add_argument("--principal_point_X", default=1, type=int, help="principal point (x)")
    parser.add_argument("--principal_point_Y", default=1, type=int, help="principal point (y)")
    parser.add_argument("--focal_const", default=1, type=int, help="a constant for normalizing different focal lengths")
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")
    parser.add_argument("--test_size", default=0, type=int, help="test set ratio")
    parser.add_argument("--root", default=os.path.join(os.getcwd(), 'data'), type=str, help='root directory')
    parser.add_argument("--r_thres", default=1, type=float, help='a threshold for successful rotation prediction')
    parser.add_argument("--t_thres", default=0.2, type=float, help='a threshold for successful translation prediction')
    parser.add_argument("--lambda", default=1, type=float, help='a hyperparameter to adjust translation scale')

    args = parser.parse_args()
    main(args)
    