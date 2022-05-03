terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "3.26.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "3.0.1"
    }
  }
  required_version = ">= 1.1.0"

  cloud {
    organization = "CYTech"

    workspaces {
      name = "soundGen-cytech"
    }
  }
}

provider "aws" {
  region = "eu-west-1"
}

resource "random_pet" "sg" {}


resource "aws_instance" "web" {
  ami                    = "ami-015ba8cf6eb94ee23"
  instance_type          = "g4dn.xlarge"
  key_name               = "ML_ICC_key"
  vpc_security_group_ids = [aws_security_group.web-sg.id]

  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y git 
              EOF
}

resource "aws_security_group" "web-sg" {
  name = "${random_pet.sg.id}-sg"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  // connectivity to ubuntu mirrors is required to run `apt-get update` and `apt-get install apache2`
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

output "gpu-instance-address" {
  value = aws_instance.web.public_dns
}